import sys
sys.path.append("../")
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit.primitives import Sampler
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from grover_optimizer import GroverOptimizer
from qiskit_optimization.translators import from_docplex_mp
from docplex.mp.vartype import BinaryVarType
from docplex.mp.model import Model

from qubo.qubo import Qubo
from models.customgraph import CustomGraph

from time import time
import numpy as np

SIMULATIONS = 50
dir_path = 'raw_data/'

def main(verbose = False):
    graph = CustomGraph(
        n_columns=2,
        n_sensor_rows=1,
        n_street_points_rows=1,
        max_sensors_radius=2,
    )
    # print starting graph
    if verbose:
        graph.plot()

    # create the qubo model
    n_sensors, n_street_points, edges = graph.get_data_for_qubo()
    qubo = Qubo(n_sensors, n_street_points, edges, penalty_coefficient=2)
    qubo_dim = len(qubo.matrix)

    # create the quadratic program with all the binary variables (named by default x0, x1, ..., xn)
    bin_vars = np.empty(qubo_dim, dtype=BinaryVarType)
    model = Model(name="min_sensor_number")
    for i in range(qubo_dim):
        bin_vars[i] = model.binary_var()

    # create the qubo expression and convert it to be used by the
    # Grover optimizer and the exact solver
    qubo_expression = bin_vars.T @ qubo.matrix @ bin_vars
    if verbose:
        print(qubo_expression)

    model.minimize(qubo_expression)
    qp = from_docplex_mp(model)
    if verbose:
        print(qp.prettyprint())

    # exact solver to check solution
    if verbose:
        exact_solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
        exact_result = exact_solver.solve(qp)
        print("Exact result: \n", exact_result.prettyprint())

        # plot the exact solution
        active_sensors = []
        variables_dict = exact_result.variables_dict
        for i in range(len(variables_dict)):
            var_key = f'x{i}'
            var_val = variables_dict[var_key]
            if var_val == 1 and i < n_sensors:
                active_sensors.append(i)
        print(active_sensors)

        graph.add_active_sensors(active_sensors)
        graph.plot()

    # Grover optimizer
    grover_result = simulate_iter_GAS(qp, qubo_dim)
    grover_result = simulate_enc_GAS(qp, qubo_dim)

    # plot the last Grover solution
    if verbose:
        active_sensors = []
        variables_dict = grover_result.variables_dict
        for i in range(len(variables_dict)):
            var_key = f'x{i}'
            var_val = variables_dict[var_key]
            if var_val == 1 and i < n_sensors:
                active_sensors.append(i)

        print(active_sensors)
        graph.add_active_sensors(active_sensors)
        graph.plot()


def run_GAS(num_encoding_qubits: int, num_iterations: int, qp: QuadraticProgram):
    grover_optimizer = GroverOptimizer(num_value_qubits=num_encoding_qubits,
                                       num_iterations=num_iterations,
                                       sampler=Sampler())
    results = grover_optimizer.solve(qp)
    return results


def simulate_enc_GAS(qp: QuadraticProgram, qubo_dim: int):
    # num_iterations is the maximum number of Grover applications with NO IMPROVEMENT
    iterations = 9

    with open(f"{dir_path}simulations_4var_9iter_enc_qubit.csv", "a") as file:
        for n in range(2, qubo_dim + 6):
            # num_iterations is the maximum number of Grover applications with NO IMPROVEMENT
            start = time()
            results = run_GAS(num_encoding_qubits=n, num_iterations=iterations, qp=qp)
            end = time()
            file.write(f'{n},\t{results.variables_dict},\t{results.fval},\t{end - start}\n')

    # return last solution
    return results

def simulate_iter_GAS(qp: QuadraticProgram, qubo_dim: int):
    # num_iterations is the maximum number of Grover applications with NO IMPROVEMENT
    num_encoding_qubits = qubo_dim

    with open(f"{dir_path}simulations_4var_iterations.csv", "a") as file:
        for iterations in [*range(1, 10),*range(10, 100, 10)]:
            # num_iterations is the maximum number of Grover applications with NO IMPROVEMENT
            start = time()
            results = run_GAS(num_encoding_qubits=num_encoding_qubits, num_iterations=iterations, qp=qp)
            end = time()
            file.write(f'{num_encoding_qubits},\t{iterations},\t{results.variables_dict},\t{results.fval},\t{end - start}\n')

    # return last solution
    return results

if __name__ == "__main__":
    print("Running Grover Optimizer test")
    # with open(f"{dir_path}simulations_4var_rotations.csv", "a") as file:
    #     file.write(f'encoding qubits,\t number of iterations,\t rotations,\t eigenvalue\n')
    # with open(f"{dir_path}simulations_4var_iterations.csv", "a") as file:
    #     file.write(f'encoding qubits,\t number of iterations,\t solution,\t eigenvalue,\t time\n')
    # with open(f"{dir_path}simulations_4var_9iter_enc_qubit.csv", "a") as file:
    #     file.write(f'encoding qubits,\t solution,\t eigenvalue,\t time\n')

    for s in range(SIMULATIONS):
        print('Simulation #', s, '\n')
        main(verbose=False)
