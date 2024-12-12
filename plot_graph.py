from math import sqrt

import matplotlib.pyplot as plt
import networkx as nx

node_attribute = "type"


def create_graph(n_street_columns: int,
                 n_street_rows: int,
                 n_sensor_columns: int,
                 n_sensor_rows: int,
                 max_sensors_radius: float):
    # Maximum radius is the diagonal of the graph which is 1x1
    # TODO: che schifo figa tutta merda
    normalized_sensor_radius = max_sensors_radius / sqrt(n_sensor_columns ** 2 + n_sensor_rows ** 2)
    print(normalized_sensor_radius)
    g = nx.Graph()
    street_point_color = "green"
    sensor_color = "red"
    colors_list = []
    nodes_position = {}
    for i in range(n_street_columns):
        for j in range(n_street_rows):
            node_index = j * n_street_columns + i
            g.add_node(node_index)
            colors_list.append(street_point_color)
            nodes_position[node_index] = [i / n_street_columns, j / n_street_rows]

    for i in range(n_sensor_columns):
        for j in range(n_sensor_rows):
            node_index = n_street_columns * n_street_rows + j * n_sensor_columns + i
            g.add_node(node_index)
            colors_list.append(sensor_color)
            nodes_position[node_index] = [i / n_sensor_columns, j / n_sensor_rows]

            for h in range(n_street_columns):
                for l in range(n_street_rows):
                    # TODO: che schifo
                    if normalized_sensor_radius > sqrt((h / n_sensor_columns - i / n_sensor_columns) ** 2 + (
                            l / n_sensor_rows - j / n_street_rows) ** 2):
                        print("Adding edge")
                        g.add_edge(node_index, node_index - n_street_columns * n_street_rows)

    return g, nodes_position, colors_list


G, positions, colors = create_graph(3,
                                    5,
                                    4,
                                    3,
                                    3)
print(G.nodes, len(colors), len(positions))
nx.draw(G,
        node_color=colors,
        pos=positions)

print(positions)
plt.show()
