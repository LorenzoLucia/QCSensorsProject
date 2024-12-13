from enum import Enum
from math import sqrt

from networkx import Graph


class NodeType(Enum):
    SENSOR = 0
    STREET_POINT = 1


class CustomGraph(Graph):
    def __init__(self,
                 n_columns: int = 1,
                 n_street_rows: int = 1,
                 n_sensor_rows: int = 1,
                 max_sensors_radius: float = 1,
                 street_point_color: str = "green",
                 sensor_color: str = "blue",
                 active_sensor_color: str = "red",
                 **attr):
        super().__init__(None, **attr)
        self.__colors = []
        self.__nodes_dict = {}

        self.n_columns = n_columns
        self.n_street_rows = n_street_rows
        self.n_sensor_rows = n_sensor_rows

        self.max_sensors_radius = max_sensors_radius

        self.street_point_color = street_point_color
        self.sensor_color = sensor_color
        self.active_sensor_color = active_sensor_color

        for i in range(self.n_columns):
            for j in range(self.n_street_rows):
                node_index = j * n_columns + i
                self.add_node(node_index)
                self.__colors.append(street_point_color)
                self.__nodes_dict[node_index] = {
                    "type": NodeType.STREET_POINT,
                    "x": i,
                    "y": j,
                }

        for i in range(self.n_columns):
            for j in range(self.n_sensor_rows):
                node_index = self.n_columns * self.n_street_rows + j * self.n_columns + i
                self.add_node(node_index)
                self.__colors.append(sensor_color)
                self.__nodes_dict[node_index] = {
                    "type": NodeType.SENSOR,
                    "x": i,
                    "y": j + 0.5,
                    "visible_street_points": []
                }

                for m in range(self.n_columns):
                    for n in range(self.n_street_rows):
                        street_point_index = n * self.n_columns + m
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
        return dict(map(lambda item: (item[0], [item[1]["x"], item[1]["y"]]), self.__nodes_dict.items()))

    def get_colors(self):
        return self.__colors
