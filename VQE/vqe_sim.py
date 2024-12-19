import sys
sys.path.append("../")
from qubo.qubo import Qubo
from models.customgraph import CustomGraph
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import to_ising
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Estimator
from qiskit.circuit.library import RealAmplitudes

from qiskit_aer import Aer


def main(entanglement='sca', reps=1, maxiter=100):
    graph = CustomGraph(
        n_columns=3,
        n_sensor_rows=2,
        n_street_points_rows=4,
        max_sensors_radius=2,
    )
    n_sensors, n_street_points, edges = graph.get_data_for_qubo()
    qubo = Qubo(n_sensors, n_street_points, edges)

    qp = QuadraticProgram()

    for i in range(len(qubo.matrix)):
        qp.binary_var()

    # qp.minimize(constant=0, quadratic=qubo.get_matrix_dict())
    qp.maximize(constant=0, quadratic=qubo.get_matrix_dict())

    operator, offset = to_ising(qp)

    optimizer = COBYLA(maxiter=maxiter)

    ansatz = RealAmplitudes(num_qubits=len(qubo.matrix), entanglement=entanglement, reps=reps)
    vqe = VQE(ansatz=ansatz, optimizer=optimizer, estimator=Estimator())
    result = vqe.compute_minimum_eigenvalue(operator)

    backend = Aer.get_backend('qasm_simulator')
    qc = ansatz.assign_parameters(result.optimal_parameters).decompose()
    print(qc)
    qc.measure_all()

    # transpiled_qc = transpile(qc, backend)
    # qobj = assemble(transpiled_qc)
    result = backend.run(qc).result()

    counts = result.get_counts()
    max_value = max(counts.values())
    for k in counts.keys():
        if counts[k] == max_value:
            max_key = k
            break
    print('Risultati delle misurazioni:', max_key, max_value)

    active_sensors = []
    for i in range(n_sensors):
        if max_key[i] == '1':
            active_sensors.append(i)
    print(active_sensors)

    graph.add_active_sensors(active_sensors)

    graph.plot()


if __name__ == '__main__':
    main(entanglement='linear', reps=2, maxiter=100)
