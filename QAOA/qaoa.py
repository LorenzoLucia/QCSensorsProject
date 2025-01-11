import time
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import Aer
from scipy.optimize import minimize

class Qaoa:
    def __init__(self, n_sensors, n_street_points, edges, p=1, backend=None):
        """
        Initialize the QAOA class.

        :param n_sensors: Number of sensors.
        :param n_street_points: Number of street points.
        :param edges: List of edges for the QUBO problem.
        :param p: Number of QAOA layers.
        :param backend: Backend to execute the quantum circuit.
        """
        self.n_sensors = n_sensors
        self.n_street_points = n_street_points
        self.edges = edges
        self.p = p
        self.num_qubits = n_sensors + n_street_points
        self.backend = backend if backend else Aer.get_backend('qasm_simulator')
        self.gamma_params = [Parameter(f'γ_{i}') for i in range(p)]
        self.beta_params = [Parameter(f'β_{i}') for i in range(p)]
        self.final_state = None

    def build_circuit(self, params):
        """
        Build the QAOA circuit based on the parameters.
        """
        qc = QuantumCircuit(self.num_qubits)
        gamma = params[:self.p]
        beta = params[self.p:]

        qc.h(range(self.num_qubits))  

        for i in range(self.p):
            # applico hamiltoniana del costo
            self._apply_cost_hamiltonian(qc, gamma[i])
            # Applico mixer hamiltoniana
            self._apply_mixer_hamiltonian(qc, beta[i])

        qc.measure_all()  
        return qc

    def _apply_cost_hamiltonian(self, circuit, gamma):
        """
        Apply the cost Hamiltonian for the QAOA circuit.
        """
        for i, j in self.edges:
            circuit.cx(i, j)
            circuit.rz(2 * gamma, j)
            circuit.cx(i, j)

    def _apply_mixer_hamiltonian(self, circuit, beta):
        """
        Apply the mixer Hamiltonian (X-gates) for the QAOA circuit.
        """
        for i in range(self.num_qubits):
            circuit.rx(2 * beta, i)

    def objective_function(self, params):
        """
        The objective function to minimize. It computes the cost of the solution
        for a given set of parameters (gamma and beta).
        """
        qc = self.build_circuit(params)
        transpiled_qc = transpile(qc, self.backend)
        job = self.backend.run(transpiled_qc, shots=1024)
        result = job.result()
        counts = result.get_counts()
        self.final_state = max(counts, key=counts.get)

        avg_cut = 0
        for bitstring, count in counts.items():
            cut = 0
            # Count active sensors (bits in the first n_sensors qubits)
            num_active_sensors = sum(int(bitstring[i]) for i in range(self.n_sensors))
            penalty = 2 * num_active_sensors
            # Count the cuts in the edges
            for i, j in self.edges:
                if bitstring[i] != bitstring[j]:
                    cut += 1
            avg_cut += (cut-penalty) * count / 1024  # costo
        return -avg_cut  # nego per minimizzazione

    def optimize(self, maxiter=50):
        """
        Optimize the QAOA parameters using COBYLA optimizer.
        """
        initial_params = np.random.rand(2 * self.p)  # Inizializzo 2p valori: prima metà per gamma e seconda metà per beta
        start_time = time.time()
        result = minimize(self.objective_function, initial_params, method='COBYLA', options={'maxiter': maxiter})
        exec_time = time.time() - start_time
        final_params = result.x
        return final_params, self.objective_function(final_params), exec_time

    def get_final_sensor_configuration(self):
        """
        Extract the final configuration of the sensors from the bitstring of the final state.
        """
        if self.final_state is None:
            return []
        else:
            return [i for i in range(self.n_sensors) if self.final_state[i] == '1']


    def run_iterations(self, iterations, solutions):
        """
        Esegui l'ottimizzazione più volte e calcola l'accuratezza basandosi
        sulla presenza di sensori corretti nella configurazione trovata.

        :param iterations: Numero di iterazioni.
        :param solutions: Lista delle configurazioni ottimali di sensori.
        :return: accuracy, avg_time, std_dev_time
        """
        results = {}
        total_time = 0
        std_time = []

        # Esegui le iterazioni
        for _ in range(iterations):
            # Ottimizzazione con QAOA
            params, _, exec_time = self.optimize()
            final_config = self.get_final_sensor_configuration()
            total_time += exec_time
            std_time.append(exec_time)

            # Conta quante volte appare una configurazione trovata
            final_config_tuple = tuple(final_config)
            if final_config_tuple not in results:
                results[final_config_tuple] = 0
            results[final_config_tuple] += 1

        # Calcolo dell'accuratezza
        total_overlap = 0
        for solution in solutions:  # Itera sulle configurazioni ottimali
            for config in results:  # Itera sulle configurazioni trovate
                # Conta la sovrapposizione con la configurazione ottimale
                intersection = len(set(config).intersection(set(solution))) / len(set(solutions))  
                total_overlap += intersection * results[config] # tengo conto del numero di volte che appare una configurazione

        # Calcola l'accuratezza come rapporto tra sensori corretti e sensori trovati (normalizzo per il numero di iterazioni)
        accuracy = total_overlap / iterations

        # Tempo medio e deviazione standard
        avg_time = total_time / iterations

        return accuracy, avg_time

