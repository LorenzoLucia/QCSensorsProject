# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2020, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""GroverOptimizer module"""

import logging
import math
from copy import deepcopy
from typing import Dict, List, Optional, Union, cast

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QuadraticForm
from qiskit.primitives import BaseSampler
from qiskit_algorithms import AmplificationProblem
from qiskit_algorithms.amplitude_amplifiers.grover import Grover
from qiskit_algorithms.utils import algorithm_globals

from qiskit_optimization.algorithms.optimization_algorithm import (
    OptimizationAlgorithm,
    OptimizationResult,
    OptimizationResultStatus,
    SolutionSample,
)
from qiskit_optimization.converters import QuadraticProgramConverter, QuadraticProgramToQubo
from qiskit_optimization.exceptions import QiskitOptimizationError
from qiskit_optimization.problems import QuadraticProgram, Variable

logger = logging.getLogger(__name__)
dir_path = 'raw_data/'

class GroverOptimizer(OptimizationAlgorithm):
    """Uses Grover Adaptive Search (GAS) to find the minimum of a QUBO function."""

    def __init__(
        self,
        num_value_qubits: int,
        num_iterations: int = 3,
        converters: Optional[
            Union[QuadraticProgramConverter, List[QuadraticProgramConverter]]
        ] = None,
        penalty: Optional[float] = None,
        sampler: Optional[BaseSampler] = None,
    ) -> None:
        """
        Args:
            num_value_qubits: The number of value qubits.
            num_iterations: The number of iterations the algorithm will search with
                no improvement.
            converters: The converters to use for converting a problem into a different form.
                By default, when None is specified, an internally created instance of
                :class:`~qiskit_optimization.converters.QuadraticProgramToQubo` will be used.
            penalty: The penalty factor used in the default
                :class:`~qiskit_optimization.converters.QuadraticProgramToQubo` converter
            sampler: A Sampler to use for sampling the results of the circuits.

        Raises:
            ValueError: If both a quantum instance and sampler are set.
            TypeError: When there one of converters is an invalid type.
        """
        self._num_value_qubits = num_value_qubits
        self._num_key_qubits = 0
        self._n_iterations = num_iterations
        self._circuit_results = {}  # type: dict
        self._converters = self._prepare_converters(converters, penalty)
        self._sampler = sampler

    def get_compatibility_msg(self, problem: QuadraticProgram) -> str:
        """Checks whether a given problem can be solved with this optimizer.

        Checks whether the given problem is compatible, i.e., whether the problem can be converted
        to a QUBO, and otherwise, returns a message explaining the incompatibility.

        Args:
            problem: The optimization problem to check compatibility.

        Returns:
            A message describing the incompatibility.
        """
        return QuadraticProgramToQubo.get_compatibility_msg(problem)

    def _get_a_operator(self, qr_key_value, problem):
        quadratic = problem.objective.quadratic.to_array()
        linear = problem.objective.linear.to_array()
        offset = problem.objective.constant

        # Get circuit requirements from input.
        quadratic_form = QuadraticForm(
            self._num_value_qubits, quadratic, linear, offset, little_endian=False
        )

        a_operator = QuantumCircuit(qr_key_value)
        a_operator.h(list(range(self._num_key_qubits)))
        a_operator.compose(quadratic_form, inplace=True)
        return a_operator

    def _get_oracle(self, qr_key_value):
        # Build negative value oracle O.
        if qr_key_value is None:
            qr_key_value = QuantumRegister(self._num_key_qubits + self._num_value_qubits)

        oracle_bit = QuantumRegister(1, "oracle")
        oracle = QuantumCircuit(qr_key_value, oracle_bit)
        oracle.z(self._num_key_qubits)  # recognize negative values.

        def is_good_state(measurement):
            """Check whether ``measurement`` is a good state or not."""
            value = measurement[
                self._num_key_qubits : self._num_key_qubits + self._num_value_qubits
            ]
            return value[0] == "1"

        return oracle, is_good_state

    def solve(self, problem: QuadraticProgram) -> OptimizationResult:
        """Tries to solve the given problem using the grover optimizer.

        Runs the optimizer to try to solve the optimization problem. If the problem cannot be,
        converted to a QUBO, this optimizer raises an exception due to incompatibility.

        Args:
            problem: The problem to be solved.

        Returns:
            The result of the optimizer applied to the problem.

        Raises:
            ValueError: If a quantum instance or a sampler has not been provided.
            ValueError: If both a quantum instance and sampler are set.
            AttributeError: If the quantum instance has not been set.
            QiskitOptimizationError: If the problem is incompatible with the optimizer.
        """
        if self._sampler is None:
            raise ValueError("A sampler must be provided.")

        self._verify_compatibility(problem)

        # convert problem to minimization QUBO problem
        problem_ = self._convert(problem, self._converters)
        problem_init = deepcopy(problem_)

        self._num_key_qubits = len(problem_.objective.linear.to_array())

        # Variables for tracking the optimum.
        optimum_found = False
        optimum_key = math.inf
        optimum_value = math.inf
        threshold = 0
        n_key = self._num_key_qubits
        n_value = self._num_value_qubits

        # Variables for tracking the solutions encountered.
        num_solutions = 2**n_key
        keys_measured = []

        # Variables for result object.
        operation_count = {}
        iteration = 0

        # Variables for stopping if we've hit the rotation max.
        rotations = 0
        max_rotations = int(np.ceil(100 * np.pi / 4))

        # Initialize oracle helper object.
        qr_key_value = QuantumRegister(self._num_key_qubits + self._num_value_qubits)
        orig_constant = problem_.objective.constant
        measurement = True
        oracle, is_good_state = self._get_oracle(qr_key_value)

        while not optimum_found:
            m = 1
            improvement_found = False

            # Get oracle O and the state preparation operator A for the current threshold.
            problem_.objective.constant = orig_constant - threshold
            a_operator = self._get_a_operator(qr_key_value, problem_)

            # Iterate until we measure a negative.
            loops_with_no_improvement = 0
            while not improvement_found:
                # Determine the number of rotations.
                loops_with_no_improvement += 1
                rotation_count = algorithm_globals.random.integers(0, m)
                rotations += rotation_count
                # Apply Grover's Algorithm to find values below the threshold.
                amp_problem = AmplificationProblem(
                    oracle=oracle,
                    state_preparation=a_operator,
                    is_good_state=is_good_state,
                )
                grover = Grover()
                circuit = grover.construct_circuit(
                    problem=amp_problem, power=rotation_count, measurement=measurement
                )

                # Get the next outcome.
                outcome = self._measure(circuit)
                k = int(outcome[0:n_key], 2)
                v = outcome[n_key : n_key + n_value]
                int_v = self._bin_to_int(v, n_value) + threshold
                logger.info("Outcome: %s", outcome)
                logger.info("Value Q(x): %s", int_v)
                # If the value is an improvement, we update the iteration parameters (e.g. oracle).
                if int_v < optimum_value:
                    optimum_key = k
                    optimum_value = int_v
                    logger.info("Current Optimum Key: %s", optimum_key)
                    logger.info("Current Optimum Value: %s", optimum_value)
                    improvement_found = True
                    threshold = optimum_value

                    # trace out work qubits and store samples
                    if self._sampler is not None:
                        self._circuit_results = {
                            i[-1 * n_key :]: v for i, v in self._circuit_results.items()
                        }
                    else:
                        self._circuit_results = {
                            i[-1 * n_key :]: v for i, v in self._circuit_results.items()
                        }
                    raw_samples = self._eigenvector_to_solutions(
                        self._circuit_results, problem_init
                    )
                    raw_samples.sort(key=lambda x: x.fval)
                    samples, _ = self._interpret_samples(problem, raw_samples, self._converters)
                else:
                    # Using Durr and Hoyer method, increase m.
                    m = int(np.ceil(min(m * 8 / 7, 2 ** (n_key / 2))))
                    logger.info("No Improvement. M: %s", m)

                    # Check if we've already seen this value.
                    if k not in keys_measured:
                        keys_measured.append(k)

                    # Assume the optimal if any of the stop parameters are true.
                    if (
                        loops_with_no_improvement >= self._n_iterations
                        or len(keys_measured) == num_solutions
                        or rotations >= max_rotations
                    ):
                        improvement_found = True
                        optimum_found = True

                # Track the operation count.
                operations = circuit.count_ops()
                operation_count[iteration] = operations
                iteration += 1
                logger.info("Operation Count: %s\n", operations)

        # If the constant is 0 and we didn't find a negative, the answer is likely 0.
        if optimum_value >= 0 and orig_constant == 0:
            optimum_key = 0

        opt_x = np.array([1 if s == "1" else 0 for s in f"{optimum_key:{n_key}b}"])
        # Compute function value of minimization QUBO
        fval = problem_init.objective.evaluate(opt_x)

        with open(f"{dir_path}simulations_{n_key}var_rotations.csv", "a") as file:
            file.write(f'{n_value},\t {self._n_iterations},\t {rotations},\t {fval}\n')

        # cast binaries back to integers and eventually minimization to maximization
        return cast(
            GroverOptimizationResult,
            self._interpret(
                x=opt_x,
                converters=self._converters,
                problem=problem,
                result_class=GroverOptimizationResult,
                samples=samples,
                raw_samples=raw_samples,
                operation_counts=operation_count,
                n_input_qubits=n_key,
                n_output_qubits=n_value,
                intermediate_fval=fval,
                threshold=threshold,
            ),
        )

    def _measure(self, circuit: QuantumCircuit) -> str:
        """Get probabilities from the given backend, and picks a random outcome."""
        probs = self._get_prob_dist(circuit)
        logger.info("Frequencies: %s", probs)
        p = list(probs.values())
        p = [p[i] / sum(p) for i in range(len(p))]
        # Pick a random outcome.
        return algorithm_globals.random.choice(list(probs.keys()), 1, p=p)[0]

    def _get_prob_dist(self, qc: QuantumCircuit) -> Dict[str, float]:
        """Gets probabilities from a given backend."""
        # Execute job and filter results.
        job = self._sampler.run([qc])

        try:
            result = job.result()
        except Exception as exc:
            print(exc)
            raise QiskitOptimizationError("Sampler job failed.") from exc
        quasi_dist = result.quasi_dists[0]
        raw_prob_dist = {
            k: v
            for k, v in quasi_dist.binary_probabilities(qc.num_qubits).items()
            if v >= self._MIN_PROBABILITY
        }
        prob_dist = {k[::-1]: v for k, v in raw_prob_dist.items()}
        self._circuit_results = {i: v**0.5 for i, v in raw_prob_dist.items()}
        return prob_dist

    @staticmethod
    def _bin_to_int(v: str, num_value_bits: int) -> int:
        """Converts a binary string of n bits using two's complement to an integer."""
        if v.startswith("1"):
            int_v = int(v, 2) - 2**num_value_bits
        else:
            int_v = int(v, 2)

        return int_v


class GroverOptimizationResult(OptimizationResult):
    """A result object for Grover Optimization methods."""

    def __init__(
        self,
        x: Union[List[float], np.ndarray],
        fval: float,
        variables: List[Variable],
        operation_counts: Dict[int, Dict[str, int]],
        n_input_qubits: int,
        n_output_qubits: int,
        intermediate_fval: float,
        threshold: float,
        status: OptimizationResultStatus,
        samples: Optional[List[SolutionSample]] = None,
        raw_samples: Optional[List[SolutionSample]] = None,
    ) -> None:
        """
        Constructs a result object with the specific Grover properties.

        Args:
            x: The solution of the problem
            fval: The value of the objective function of the solution
            variables: A list of variables defined in the problem
            operation_counts: The counts of each operation performed per iteration.
            n_input_qubits: The number of qubits used to represent the input.
            n_output_qubits: The number of qubits used to represent the output.
            intermediate_fval: The intermediate value of the objective function of the
                minimization qubo solution, that is expected to be consistent to ``fval``.
            threshold: The threshold of Grover algorithm.
            status: the termination status of the optimization algorithm.
            samples: the x values, the objective function value of the original problem,
                the probability, and the status of sampling.
            raw_samples: the x values of the QUBO, the objective function value of the
                minimization QUBO, and the probability of sampling.
        """
        super().__init__(
            x=x,
            fval=fval,
            variables=variables,
            status=status,
            raw_results=None,
            samples=samples,
        )
        self._raw_samples = raw_samples
        self._operation_counts = operation_counts
        self._n_input_qubits = n_input_qubits
        self._n_output_qubits = n_output_qubits
        self._intermediate_fval = intermediate_fval
        self._threshold = threshold

    @property
    def operation_counts(self) -> Dict[int, Dict[str, int]]:
        """Get the operation counts.

        Returns:
            The counts of each operation performed per iteration.
        """
        return self._operation_counts

    @property
    def n_input_qubits(self) -> int:
        """Getter of n_input_qubits

        Returns:
            The number of qubits used to represent the input.
        """
        return self._n_input_qubits

    @property
    def n_output_qubits(self) -> int:
        """Getter of n_output_qubits

        Returns:
            The number of qubits used to represent the output.
        """
        return self._n_output_qubits

    @property
    def intermediate_fval(self) -> float:
        """Getter of the intermediate fval

        Returns:
            The intermediate value of fval before interpret.
        """
        return self._intermediate_fval

    @property
    def threshold(self) -> float:
        """Getter of the threshold of Grover algorithm.

        Returns:
            The threshold of Grover algorithm.
        """
        return self._threshold

    @property
    def raw_samples(self) -> Optional[List[SolutionSample]]:
        """Returns the list of raw solution samples of ``GroverOptimizer``.

        Returns:
            The list of raw solution samples of ``GroverOptimizer``.
        """
        return self._raw_samples
