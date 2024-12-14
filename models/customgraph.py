from math import sqrt
from typing import Union

import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from networkx import Graph

from models.node import NodeType, StreetPointNode, SensorNode


class CustomGraph(Graph):
    """
    n_columns: number of columns in the graph
    n_street_points_rows: number of street points rows in the graph
    n_sensors_rows: number of sensors rows in the graph
    max_sensor_radius: maximum distance where a sensor can cover a street point, street points closer than this
        value from a sensor will be connected to it with an edge
    street_point_color: color of the street point in the graph plotting
    sensor_color: color of the sensor in the graph plotting
    active_sensor_color: color of a sensor that has been activated in graph plotting

    """

    def __init__(self,
                 n_columns: int = 1,
                 n_sensor_rows: int = 1,
                 n_street_points_rows: int = 1,
                 max_sensors_radius: float = 1,
                 street_point_color: str = "green",
                 sensor_color: str = "blue",
                 active_sensor_color: str = "red",
                 **attr):
        super().__init__(None, **attr)

        self.n_columns = n_columns
        self.n_sensor_rows = n_sensor_rows
        self.n_street_points_rows = n_street_points_rows

        self.__nodes_dict: dict[int, Union[SensorNode, StreetPointNode]] = {}

        self.max_sensors_radius = max_sensors_radius

        self.street_point_color = street_point_color
        self.sensor_color = sensor_color
        self.active_sensor_color = active_sensor_color

        for i in range(self.n_columns):
            for j in range(self.n_street_points_rows):
                node_index = self.n_columns * self.n_sensor_rows + j * n_columns + i
                self.add_node(node_index)
                self.__nodes_dict[node_index] = StreetPointNode(i, j)

        for i in range(self.n_columns):
            for j in range(self.n_sensor_rows):
                node_index = j * self.n_columns + i
                self.add_node(node_index)
                self.__nodes_dict[node_index] = SensorNode(i, j + 0.5, active=False)

                for m in range(self.n_columns):
                    for n in range(self.n_street_points_rows):
                        street_point_index = self.n_columns * self.n_sensor_rows + n * self.n_columns + m
                        node_distance = self.__calc_node_distance(self.__nodes_dict[node_index],
                                                                  self.__nodes_dict[street_point_index])
                        if self.max_sensors_radius > node_distance:

                            self.add_edge(node_index, street_point_index, weight=round(node_distance, 1))
                            sensor = self.__nodes_dict[node_index]

                            if isinstance(sensor, SensorNode):
                                sensor.add_visible_street_point(
                                    street_point_index,
                                    self.__nodes_dict[street_point_index].x,
                                    self.__nodes_dict[street_point_index].y)
                            else:
                                raise Exception(f"Node {sensor.x, sensor.y} is not a sensor")

    @staticmethod
    def __calc_node_distance(sensor: SensorNode,
                             street_point: StreetPointNode):
        return sqrt(
            (sensor.x - street_point.x) ** 2 + (sensor.y - street_point.y) ** 2
        )

    def __get_positions(self):
        """
        :return: dictionary with as key the index of the node and as value an array with the x and y coordinate of the
        node
        """
        return dict(map(lambda item: (item[0], [item[1].x, item[1].y]), self.__nodes_dict.items()))

    def get_data_for_qubo(self):
        n_sensors = self.n_sensor_rows * self.n_columns
        n_street_points = self.n_street_points_rows * self.n_columns
        edges = []
        for i in self.__nodes_dict.keys():
            if self.__nodes_dict[i].node_type == NodeType.SENSOR:
                for j in self.__nodes_dict[i].__visible_street_points:
                    edges.append({i, j["street_point"]})
        return n_sensors, n_street_points, edges

    def add_active_sensors(self, active_sensors: list[int]):
        for i in active_sensors:
            if self.__nodes_dict[i].node_type == NodeType.SENSOR:
                self.__nodes_dict[i].active = True

    def plot(self):
        positions = self.__get_positions()
        inactive_sensors = []
        active_sensors = []
        street_points = []
        edges = []
        for i in self.__nodes_dict.keys():
            node = self.__nodes_dict[i]
            if node.node_type == NodeType.SENSOR:

                for j in node.__visible_street_points:
                    edges.append((i, j["street_point"]))

                if node.active:
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

        edge_labels = nx.get_edge_attributes(self, "weight")
        nx.draw_networkx_edge_labels(self, positions, edge_labels)

        legend_elements = [
            Line2D([0],
                   [0],
                   marker='o',
                   color='w',
                   label='Street point to cover',
                   markerfacecolor=self.street_point_color,
                   markersize=10),
            Line2D([0],
                   [0],
                   marker='o',
                   color='w',
                   label='Possible sensor positions',
                   markerfacecolor=self.sensor_color,
                   markersize=10),
            Line2D([0],
                   [0],
                   marker='o',
                   color='w',
                   label='Active sensors',
                   markerfacecolor=self.active_sensor_color,
                   markersize=10),
        ]

        plt.legend(handles=legend_elements, loc='upper right')

        plt.show()
