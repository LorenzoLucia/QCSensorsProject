import numpy as np


class Qubo:
    def __init__(self,
                 n_sensors: int,
                 n_street_points: int,
                 edges: list[set[int, int]],
                 penalty_coefficient: int = 2):

        self.n_sensors = n_sensors
        self.n_street_points = n_street_points
        self.edges = edges
        self.penalty_coefficient = penalty_coefficient
        self.matrix = np.zeros((self.n_sensors + self.n_street_points, self.n_sensors + self.n_street_points))

        for i in range(self.n_sensors):
            self.matrix[i][i] += 1

        for i in range(self.n_street_points):
            street_point_index = self.n_sensors + i
            # Sensors and slack relative to this street point
            # Adding the slack variable relative to this street point
            sensors_and_slacks = [street_point_index]
            for edge in self.edges:
                if street_point_index in edge:
                    sensor_index = next(iter(edge - {street_point_index}))
                    sensors_and_slacks.append(sensor_index)

            for j in sensors_and_slacks:
                for h in sensors_and_slacks:
                    if j == h:
                        if j == street_point_index:
                            # Then it is the slack variable
                            self.matrix[j][h] += 3 * self.penalty_coefficient
                        else:
                            # It is not the slack, then it is a sensor (x_i)
                            self.matrix[j][h] -= self.penalty_coefficient
                    if j == street_point_index or h == street_point_index:
                        self.matrix[j][h] -= 2 * self.penalty_coefficient
                    else:
                        self.matrix[j][h] += 2 * self.penalty_coefficient

    def get_matrix_dict(self):
        matrix_dict = {}
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[i])):
                matrix_dict[(i, j)] = self.matrix[i][j]
        return matrix_dict
# def create_qubo_matrix(self.n_sensors: int, n_street_points: int, edges: list[set[int, int]], penalty_coefficient: int = 2):
#     qubo_matrix = np.zeros((self.n_sensors + n_street_points, n_sensors + n_street_points))
#     for i in range(n_sensors):
#         qubo_matrix[i][i] += 1
#
#     for i in range(n_street_points):
#         street_point_index = n_sensors + i
#         # Sensors and slack relative to this street point
#         # Adding the slack variable relative to this street point
#         sensors_and_slacks = [street_point_index]
#         for edge in edges:
#             if street_point_index in edge:
#                 sensor_index = next(iter(edge - {street_point_index}))
#                 sensors_and_slacks.append(sensor_index)
#
#         for j in sensors_and_slacks:
#             for h in sensors_and_slacks:
#                 if j == h:
#                     if j == street_point_index:
#                         # Then it is the slack variable
#                         qubo_matrix[j][h] += 3 * penalty_coefficient
#                     else:
#                         # It is not the slack, then it is a sensor (x_i)
#                         qubo_matrix[j][h] -= penalty_coefficient
#                 if j == street_point_index or h == street_point_index:
#                     qubo_matrix[j][h] -= 2 * penalty_coefficient
#                 else:
#                     qubo_matrix[j][h] += 2 * penalty_coefficient
#
#     return qubo_matrix
