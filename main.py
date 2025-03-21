from qubovert.sim import anneal_qubo

from models.customgraph import CustomGraph
from qubo.qubo import Qubo


def main():
    graph = CustomGraph(
        n_columns=3,
        n_sensor_rows=2,
        n_street_points_rows=3,
        max_sensors_radius=2,
    )

    graph.plot()

    n_sensors, n_street_points, edges = graph.get_data_for_qubo()

    print(f"Number of sensors: {n_sensors}")
    print(f"Number of street points: {n_street_points}")
    qubo = Qubo(n_sensors, n_street_points, edges)
    print(f"Formulation matrix:\n{qubo.get_matrix_dict()}")

    solution = anneal_qubo(qubo.get_matrix_dict(), num_anneals=1000)
    print(solution.best.value)
    active_sensors = []
    for i in solution.best.state.keys():
        if solution.best.state[i] == 1 and i < n_sensors:
            active_sensors.append(i)
    print(active_sensors)

    graph.add_active_sensors(active_sensors)

    graph.plot()


if __name__ == '__main__':
    main()
