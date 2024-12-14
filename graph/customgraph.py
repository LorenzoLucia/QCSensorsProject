from enum import Enum
from math import sqrt

import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from networkx import Graph


class NodeType(Enum):
    SENSOR = 0
    STREET_POINT = 1


class CustomGraph(Graph):
    """
    Creates a graph with n_columns and a different number of rows for the sensors and the street points

    It can be improved to create better graph
    """

    # TODO: magari mettere che si vedono gli street point non coperti da sensori quando si plotta il grafo

    def __init__(self,
                 n_columns: int = 1,
                 n_sensor_rows: int = 1,
                 n_street_rows: int = 1,
                 max_sensors_radius: float = 1,
                 street_point_color: str = "green",
                 sensor_color: str = "blue",
                 active_sensor_color: str = "red",
                 **attr):
        super().__init__(None, **attr)

        self.n_columns = n_columns
        self.n_sensor_rows = n_sensor_rows
        self.n_street_rows = n_street_rows

        self.__colors = [street_point_color] * (n_columns * n_sensor_rows + n_columns * n_street_rows)
        self.__nodes_dict = {}

        self.max_sensors_radius = max_sensors_radius

        self.street_point_color = street_point_color
        self.sensor_color = sensor_color
        self.active_sensor_color = active_sensor_color

        for i in range(self.n_columns):
            for j in range(self.n_street_rows):
                node_index = self.n_columns * self.n_sensor_rows + j * n_columns + i
                self.add_node(node_index)
                self.__colors[node_index] = self.street_point_color
                self.__nodes_dict[node_index] = {
                    "type": NodeType.STREET_POINT,
                    "x": i,
                    "y": j,
                }

        for i in range(self.n_columns):
            for j in range(self.n_sensor_rows):
                node_index = j * self.n_columns + i
                self.add_node(node_index)
                self.__colors[node_index] = self.sensor_color
                self.__nodes_dict[node_index] = {
                    "type": NodeType.SENSOR,
                    "x": i,
                    "y": j + 0.5,
                    "visible_street_points": [],
                    "active": False
                }

                for m in range(self.n_columns):
                    for n in range(self.n_street_rows):
                        street_point_index = self.n_columns * self.n_sensor_rows + n * self.n_columns + m
                        if self.__is_street_point_visible(self.__nodes_dict[node_index],
                                                          self.__nodes_dict[street_point_index]):
                            print("Adding edge")
                            self.add_edge(node_index, street_point_index)
                            self.__nodes_dict[node_index]["visible_street_points"].append(street_point_index)

    def __is_street_point_visible(self,
                                  sensor: dict,
                                  street_point: dict):
        return self.max_sensors_radius >= sqrt(
            (sensor["x"] - street_point["x"]) ** 2 + (sensor["y"] - street_point["y"]) ** 2
        )

    def get_positions(self):
        """
        :return: dictionary with as key the index of the node and as value an array with the x and y coordinate of the
        node
        """
        return dict(map(lambda item: (item[0], [item[1]["x"], item[1]["y"]]), self.__nodes_dict.items()))

    def get_colors(self):
        colors_dict = {}
        for i in range(len(self.__colors)):
            colors_dict[i] = self.__colors[i]
        return colors_dict

    def get_data_for_qubo(self):
        n_sensors = self.n_sensor_rows * self.n_columns
        n_street_points = self.n_street_rows * self.n_columns
        edges = []
        for i in self.__nodes_dict.keys():
            if self.__nodes_dict[i]["type"] == NodeType.SENSOR:
                for j in self.__nodes_dict[i]["visible_street_points"]:
                    edges.append({i, j})
        return n_sensors, n_street_points, edges

    def add_active_sensors(self, active_sensors: list[int]):
        for i in active_sensors:
            self.__colors[i] = self.active_sensor_color
            self.__nodes_dict[i]["active"] = True

    def plot(self):
        positions = self.get_positions()
        inactive_sensors = []
        active_sensors = []
        street_points = []
        edges = []
        for i in self.__nodes_dict.keys():
            if self.__nodes_dict[i]["type"] == NodeType.SENSOR:

                for j in self.__nodes_dict[i]["visible_street_points"]:
                    edges.append((i, j))

                if self.__nodes_dict[i]["active"]:
                    active_sensors.append(i)
                else:
                    inactive_sensors.append(i)

            else:
                street_points.append(i)

        print(edges)
        nx.draw_networkx_nodes(self,
                               pos=positions,
                               nodelist=inactive_sensors,
                               node_color=self.sensor_color)

        nx.draw_networkx_nodes(self,
                               pos=positions,
                               nodelist=active_sensors,
                               node_color=self.active_sensor_color)

        nx.draw_networkx_nodes(self,
                               pos=positions,
                               nodelist=street_points,
                               node_color=self.street_point_color)

        nx.draw_networkx_edges(self,
                               pos=positions,
                               edgelist=edges)
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Street point to cover', markerfacecolor='green',
                   markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Possible sensor positions', markerfacecolor='blue',
                   markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Active sensors', markerfacecolor='red', markersize=10),
        ]

        plt.legend(handles=legend_elements, loc='upper right')

        plt.show()
