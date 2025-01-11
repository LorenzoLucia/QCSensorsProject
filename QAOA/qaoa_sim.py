from qiskit_optimization import QuadraticProgram
from qubovert.sim import anneal_qubo
from models.customgraph import CustomGraph
from qaoa import Qaoa
import matplotlib.pyplot as plt
import csv
from qubo.qubo import Qubo



def main():
    graph = CustomGraph(
        n_columns=3,
        n_sensor_rows=4,
        n_street_points_rows=4,
        max_sensors_radius=2,
    )


    n_sensors, n_street_points, edges = graph.get_data_for_qubo()
    qubo = Qubo(n_sensors=n_sensors, n_street_points=n_street_points, edges=edges)
    print(f"Number of sensors: {n_sensors}")
    print(f"Number of street points: {n_street_points}")
    
    SOLUTIONS = [(4,),(1,)]
    ITERATIONS = 100


    results = []

    with open('res_file.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['n_sensors','n_street_points','p', 'params', 'cost', 'exec_time', 'final_config', 'accuracy'])
        
        for p in range(4, 6):
            qaoa = Qaoa(qubo=qubo, p=p)
            accuracy, avg_time = qaoa.run_iterations(ITERATIONS, SOLUTIONS)
            params, avg_cut, exec_time = qaoa.optimize(maxiter=100)
            
            final_config = qaoa.get_final_sensor_configuration()
            print(f"n_sensors={n_sensors},p={p}, Params={params}, Exec Time={exec_time}, Cost={avg_cut}, Final Config={final_config}, Accuracy={accuracy}")
            
            accuracy = round(accuracy,2) #salvo solo 2 cifre decimali
            results.append({'n_sensors': n_sensors, 'n_street_points': n_street_points,'p': p, 'params': params, 'cost': avg_cut, 'exec_time': avg_time, 'final_config': final_config, 'accuracy': accuracy})
            writer.writerow([n_sensors, n_street_points, p, params, avg_cut, avg_time, final_config, accuracy])
            
        

if __name__ == '__main__':
    main()
