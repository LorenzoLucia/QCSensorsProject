from enum import Enum


# TODO: cazzo inutile

class NodeType(Enum):
    SENSOR = 0
    STREET_POINT = 1


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def to_tuple(self):
        return self.x, self.y


class SensorNode(Node):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.type = NodeType.SENSOR

    def to_tuple(self):
        return self.x, self.y, self.type


class StreetPointNode(Node):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.type = NodeType.STREET_POINT

    def to_tuple(self):
        return self.x, self.y, self.type
