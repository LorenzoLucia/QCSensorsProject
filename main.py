from qubovert.sim import anneal_qubo

from graph.customgraph import CustomGraph
from qubo.qubo import Qubo


def main():
    graph = CustomGraph(
        n_columns=4,
        n_sensor_rows=2,
        n_street_rows=4,
        max_sensors_radius=2,
    )

    graph.plot()

    n_sensors, n_street_points, edges = graph.get_data_for_qubo()

    print(f"Number of sensors: {n_sensors}")
    print(f"Number of street points: {n_street_points}")
    qubo = Qubo(n_sensors, n_street_points, edges)

    solution = anneal_qubo(qubo.get_matrix_dict(), num_anneals=1000)

    active_sensors = []
    for i in solution.best.state.keys():
        if solution.best.state[i] == 1 and i < n_sensors:
            active_sensors.append(i)

    print(graph.get_positions())
    print(graph.get_colors())

    graph.add_active_sensors(active_sensors)

    graph.plot()


if __name__ == '__main__':
    main()
