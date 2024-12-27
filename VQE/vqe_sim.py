import sys
sys.path.append("../")
from qubo.qubo import Qubo
from models.customgraph import CustomGraph
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import to_ising
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import POWELL
from qiskit.primitives import Estimator
from qiskit.circuit.library import RealAmplitudes, efficient_su2
from qiskit.providers.basic_provider import BasicProvider

from time import time
import concurrent.futures
import numpy as np 

SOLUTIONS = [(4,), (1,)]
ITERATIONS = 100

def solutions_iterations(operator, ansatz, n_sensors, maxiter):
    optimizer = POWELL(maxiter=maxiter)
    backend = BasicProvider().get_backend('basic_simulator')
    vqe = VQE(ansatz=ansatz, optimizer=optimizer, estimator=Estimator())
    solutions = {}
    avg_time = 0
    std_time = []
    for _ in range(ITERATIONS):
        start = time()
        result = vqe.compute_minimum_eigenvalue(operator)
        
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
        # print(sensor_counts)

        max_value = max(sensor_counts.values())
        for k in sensor_counts.keys():
            if sensor_counts[k] == max_value:
                max_key = k
                break
        # print('Risultati delle misurazioni:', max_key, max_value)

        active_sensors = []
        for i in range(n_sensors):
            if max_key[-1-i] == '1':
                active_sensors.append(i)
        # print(active_sensors)

        active_sensors = tuple(active_sensors)
        if active_sensors not in solutions.keys():
            solutions[active_sensors] = 0
        solutions[active_sensors] += 1

        end = time()
        avg_time += (end-start)/ITERATIONS
        std_time.append(end-start)
        # print(f"Iteration {j} finished in {end-start}s")

    # outfile = open("result_powell.out", "w")

    max_value = max(solutions.values())
    for k in solutions.keys():
        if solutions[k] == max_value:
            max_key = k
        # outfile.write(f"{k}: {solutions[k]}\n")
    accuracy = 0
    for s in SOLUTIONS:
        if s in solutions.keys():
            accuracy += float(solutions[s]/ITERATIONS)

    print('Risultati delle misurazioni:', max_key, max_value)

    return accuracy, avg_time, np.std(std_time)


def run_simulation(ansatz_type='RealAmplitudes', entanglement='sca', reps=1, maxiter=1000):
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

    if ansatz_type=='RealAmplitudes':
        ansatz = RealAmplitudes(num_qubits=len(qubo.matrix), entanglement=entanglement, reps=reps)
    elif ansatz_type == 'SU2':
        ansatz = efficient_su2(len(qubo.matrix), su2_gates=['rx', 'y'], entanglement=entanglement, reps=reps)

    print(f"Starting iterations for ", ansatz_type, entanglement, reps)
    s = time()
    accuracy, avg_time, std_dev = solutions_iterations(operator, ansatz, n_sensors, maxiter)
    e = time()
    print(accuracy, avg_time)
    print(f"Completed in {e-s}s")
    # graph.add_active_sensors(active_sensors)
    # df.loc[len(df)] = [ansatz_type, entanglement, reps, accuracy, avg_time]
    outfile = open("results_rep_raw.csv", "a")
    outfile.write(f"{ansatz_type},{entanglement},{reps},{accuracy},{avg_time},{std_dev}\n")
    outfile.close()
    # print(df)
    # graph.plot()


if __name__ == '__main__':
    # df = pd.DataFrame({'ansatz':[], 'entanglement':[], 'reps':[], 'accuracy':[], 'avg_time':[]})
    outfile = open("results_rep_raw.csv", "w")
    outfile.write(f"ansatz,entanglement,reps,accuracy,avg_time,std_dev\n")
    outfile.close()
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for ansatz in ['SU2', 'RealAmplitudes']:
            for entanglement in ['full', 'linear', 'reverse_linear', 'sca', 'circular']:
                for reps in range(1,4):
                    futures.append(executor.submit(run_simulation, ansatz_type=ansatz, entanglement=entanglement, reps=reps, maxiter=200))
        concurrent.futures.wait(futures)
    # df.to_csv("performances_comparison.csv")
