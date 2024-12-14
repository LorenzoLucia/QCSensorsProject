from enum import Enum
from math import sqrt


class NodeType(Enum):
    SENSOR = 0
    STREET_POINT = 1


class StreetPointNode:
    """
    x: position on the x-axis
    y: position on the y-axis
    node_type: type of the node
    """

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.node_type = NodeType.STREET_POINT


class SensorNode:
    """
    x: position on the x-axis
    y: position on the y-axis
    node_type: type of the node
    visible_street_points: list of street points that are visible by the sensors, therefore
        their distance is smaller than the maximum radius of the sensors
    active: whether the sensor is active

    """

    def __init__(self, x: float, y: float, visible_street_points=None, active: bool = False):
        self.x = x
        self.y = y
        self.node_type = NodeType.SENSOR
        if visible_street_points is None:
            visible_street_points = []
        self.__visible_street_points: list[dict] = visible_street_points
        self.active = active

    def __calc_distance(self, x, y):
        return sqrt(
            (self.x - x) ** 2 + (self.y - y) ** 2
        )

    def add_visible_street_point(self, index: int, x: float, y: float):
        self.__visible_street_points.append(
            {
                "street_point": index,
                "weight": self.__calc_distance(x, y),
            }
        )
