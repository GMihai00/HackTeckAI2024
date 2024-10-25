
import math

def distance_between_points(point1, point2):
    int_x = abs(point1[0] - point2[0])
    int_y = abs(point1[1] - point2[1])

    return math.sqrt(int_x ** 2 + int_y ** 2)
