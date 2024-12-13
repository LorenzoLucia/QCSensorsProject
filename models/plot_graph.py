import matplotlib.pyplot as plt
import networkx as nx

from models.customgraph import CustomGraph

# Test script to try the graph class

graph = CustomGraph(
    n_columns=3,
    n_street_rows=5,
    n_sensor_rows=4,
    n_sensor_columns=4,
    max_sensors_radius=2,
)

nx.draw(graph,
        node_color=graph.get_colors(),
        pos=graph.get_positions())

print(graph.get_colors())
print(graph.get_positions())
plt.show()
