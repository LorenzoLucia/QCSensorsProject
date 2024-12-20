import sys
sys.path.append("../")
from qubo.qubo import Qubo
from models.customgraph import CustomGraph
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import to_ising
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA, POWELL, ADAM, AQGD, GradientDescent, QNSPSA
from qiskit.primitives import Estimator
from qiskit.circuit.library import RealAmplitudes
from qiskit.providers.basic_provider import BasicProvider
from qiskit.quantum_info import Statevector

from time import time


def main(entanglement='sca', reps=1, maxiter=100):
    graph = CustomGraph(
        n_columns=3,
        n_sensor_rows=2,
        n_street_points_rows=3,
        max_sensors_radius=2,
    )
    n_sensors, n_street_points, edges = graph.get_data_for_qubo()
    qubo = Qubo(n_sensors, n_street_points, edges)

    qp = QuadraticProgram()

    for i in range(len(qubo.matrix)):
        qp.binary_var()

    # qp.minimize(constant=0, quadratic=qubo.get_matrix_dict())
    qp.minimize(constant=0, quadratic=qubo.get_matrix_dict())

    operator, offset = to_ising(qp)

    # optimizer = COBYLA(maxiter=maxiter)
    optimizer = POWELL(maxiter=maxiter)
    backend = BasicProvider().get_backend('basic_simulator')

    ansatz = RealAmplitudes(num_qubits=len(qubo.matrix), entanglement=entanglement, reps=reps)

    vqe = VQE(ansatz=ansatz, optimizer=optimizer, estimator=Estimator())
    solutions = {}
    for j in range(100):
        start = time()
        result = vqe.compute_minimum_eigenvalue(operator)
        print(result.eigenvalue)
        
        # backend = Aer.get_backend('qasm_simulator')
        
        qc = ansatz.assign_parameters(result.optimal_parameters).decompose()
        
        qc.measure_all()

        # transpiled_qc = transpile(qc, backend)
        # qobj = assemble(transpiled_qc)
        result = backend.run(qc).result()

        counts = result.get_counts()
        sensor_counts  = {}
        for k,v in counts.items():
            if k[len(k)-n_sensors:] not in sensor_counts.keys():
                sensor_counts[k[len(k)-n_sensors:]] = v
            else:
                sensor_counts[k[len(k)-n_sensors:]] += v
        print(sensor_counts)

        max_value = max(sensor_counts.values())
        for k in sensor_counts.keys():
            if sensor_counts[k] == max_value:
                max_key = k
                break
        print('Risultati delle misurazioni:', max_key, max_value)

        active_sensors = []
        for i in range(n_sensors):
            if max_key[-1-i] == '1':
                active_sensors.append(i)
        print(active_sensors)

        active_sensors = tuple(active_sensors)
        if active_sensors not in solutions.keys():
            solutions[active_sensors] = 0
        solutions[active_sensors] += 1

        end = time()

        print(f"Iteration {j} finished in {end-start}s")

    outfile = open("result_powell.out", "w")

    max_value = max(solutions.values())
    for k in solutions.keys():
        if solutions[k] == max_value:
            max_key = k
        outfile.write(f"{k}: {solutions[k]}")
        
    print('Risultati delle misurazioni:', max_key, max_value)

    outfile.close()
    # graph.add_active_sensors(active_sensors)

    # graph.plot()


if __name__ == '__main__':
    main(entanglement='full', reps=1, maxiter=100)
