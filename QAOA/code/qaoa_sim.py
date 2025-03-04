from qiskit_optimization import QuadraticProgram
from models.customgraph import CustomGraph
from qaoa import Qaoa
import matplotlib.pyplot as plt
import csv
from qubo.qubo import Qubo
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2
from qiskit_aer.noise import NoiseModel



def main():
    
    graph = CustomGraph(
        n_columns=2,
        n_sensor_rows=2,
        n_street_points_rows=2,
        max_sensors_radius=1,
    )
    
    SOLUTIONS = [(4,), (1,)]
    ITERATIONS = 100
    
    results = []
    
    with open('res_file1.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['n_sensors', 'n_street_points', 'p', 'params', 'cost', 'exec_time', 'final_config', 'accuracy', 'fidelity'])
        
        
        n_sensors, n_street_points, edges = graph.get_data_for_qubo()
        
        print(f"Testing with n_sensors={n_sensors}, n_street_points={n_street_points}")
    
        for p in range(1, 6):
            fake_backend = FakeGuadalupeV2()
            noise_model = NoiseModel.from_backend(fake_backend)
            qaoa = Qaoa(n_sensors, n_street_points, edges, p=p, backend=None, noise_model=None)
            accuracy, avg_time = qaoa.run_iterations(ITERATIONS, SOLUTIONS)
            params, avg_cut, exec_time = qaoa.optimize(maxiter=100)
            final_config = qaoa.get_final_sensor_configuration()
            fidelity = qaoa.compute_fidelity()
        
            print(f"n_sensors={n_sensors}, p={p}, Params={params}, Exec Time={exec_time}, Cost={avg_cut}, Final Config={final_config}, Accuracy={accuracy}, Fidelity = {fidelity}")
        
            accuracy = round(accuracy, 4)
            #fidelity = round(fidelity, 4)
            results.append({'n_sensors': n_sensors, 'n_street_points': n_street_points, 'p': p, 'params': params, 'cost': avg_cut, 'exec_time': avg_time, 'final_config': final_config, 'accuracy': accuracy, 'fidelity': fidelity})
            writer.writerow([n_sensors, n_street_points, p, params, avg_cut, avg_time, final_config, accuracy])

if __name__ == '__main__':
    main()
