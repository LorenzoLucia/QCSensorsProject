import time
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import Aer
from qiskit_aer.noise import NoiseModel
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import to_ising
from scipy.optimize import minimize
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2
from qubo.qubo import Qubo


def build_qubo(n_sensors, n_street_points, edges):
    qubo = Qubo(n_sensors, n_street_points, edges, penalty_coefficient=2)
    # Aggiunta delle variabili binarie
    for i in range(n_sensors + n_street_points):
        qubo.binary_var(f"x{i}")

    # Definizione della funzione obiettivo
    linear_terms = {f"x{i}": -1 for i in range(n_sensors + n_street_points)}
    quadratic_terms = {(f"x{i}", f"x{j}"): 2 for i, j in edges}
    
    qubo.minimize(linear=linear_terms, quadratic=quadratic_terms)
    
    return qubo


class Qaoa:
    def __init__(self, n_sensors, n_street_points, edges, p=1, backend=None, noise_model=None):
        self.n_sensors = n_sensors
        self.n_street_points = n_street_points
        self.edges = edges
        self.p = p
        self.num_qubits = n_sensors + n_street_points
        self.backend = backend if backend else Aer.get_backend('qasm_simulator')
        self.noise_model = noise_model
        self.gamma_params = [Parameter(f'γ_{i}') for i in range(p)]
        self.beta_params = [Parameter(f'β_{i}') for i in range(p)]
        self.final_state = None
        qubo = build_qubo(n_sensors, n_street_points, edges)
        self.qubo_hamiltonian, _ = to_ising(qubo)
        #print(self.qubo_hamiltonian)

    def build_circuit(self, params):
        qc = QuantumCircuit(self.num_qubits)
        gamma = params[:self.p]
        beta = params[self.p:]
        qc.h(range(self.num_qubits))
        for i in range(self.p):
            self._apply_cost_hamiltonian(qc, gamma[i])
            self._apply_mixer_hamiltonian(qc, beta[i])
        qc.measure_all()
        return qc

    def _apply_cost_hamiltonian(self, circuit, gamma):
        for pauli, coeff in zip(self.qubo_hamiltonian.parameters, self.qubo_hamiltonian.coeffs):
            circuit.append(pauli, list(pauli.indices))
            circuit.rz(2 * gamma * coeff, pauli.indices[-1])
            circuit.append(pauli, list(pauli.indices))

    def _apply_mixer_hamiltonian(self, circuit, beta):
        for i in range(self.num_qubits):
            circuit.rx(2 * beta, i)

    def execute_circuit(self, qc, noise_model):
        transpiled_qc = transpile(qc, self.backend)
        if noise_model:
            job = self.backend.run(transpiled_qc, shots=1024, noise_model=self.noise_model)
        else:
            job = self.backend.run(transpiled_qc, shots=1024)
        
        return job.result().get_counts()

    def objective_function(self, params):
        qc = self.build_circuit(params)
        counts = self.execute_circuit(qc, noise_model=None)
        self.final_state = max(counts, key=counts.get)
        avg_cut = 0
        for bitstring, count in counts.items():
            cut = 0
            num_active_sensors = sum(int(bitstring[i]) for i in range(self.n_sensors))
            penalty = 2 * num_active_sensors
            for i, j in self.edges:
                if bitstring[i] != bitstring[j]:
                    cut += 1
            avg_cut += (cut - penalty) * count / 1024
        return -avg_cut

    def optimize(self, maxiter=50):
        initial_params = np.random.rand(2 * self.p)
        start_time = time.time()
        result = minimize(self.objective_function, initial_params, method='COBYLA', options={'maxiter': maxiter})
        exec_time = time.time() - start_time
        return result.x, self.objective_function(result.x), exec_time
    
    
    def run_iterations(self, iterations, solutions):
    
        correct_count = 0
        total_time = 0

        for _ in range(iterations):
            params, cost, exec_time = self.optimize(maxiter=100)
            self.final_state = self.get_final_sensor_configuration()

            if self.final_state in solutions:
                correct_count += 1

            total_time += exec_time

        accuracy = correct_count / iterations
        avg_time = total_time / iterations
        return accuracy, avg_time
    
    def get_final_sensor_configuration(self):
        """
        Extract the final configuration of the sensors from the bitstring of the final state.
        """
        if self.final_state is None:
            return []
        else:
            return [i for i in range(self.n_sensors) if self.final_state[i] == '1']


    def compute_fidelity(self):
        qc_ideal = self.build_circuit(np.random.rand(2 * self.p))
        counts_ideal = self.execute_circuit(qc_ideal, noise_model=None)
        
        fake_backend = FakeGuadalupeV2()
        
        noise_model = NoiseModel.from_backend(fake_backend)
        self.noise_model = noise_model
        counts_noisy = self.execute_circuit(qc_ideal, noise_model=noise_model)
        
        fidelity = sum(min(counts_ideal.get(state, 0), counts_noisy.get(state, 0)) for state in counts_ideal) / 1024
        return fidelity

