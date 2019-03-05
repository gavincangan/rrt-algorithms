# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

import math
import numpy as np

def distance_between_points(a, b):
    """
    Return the Euclidean distance between two points
    :param a: first point
    :param b: second point
    :return: Euclidean distance between a and b
    """
    distance = sum(map(lambda a_b: (a_b[0] - a_b[1]) ** 2, zip(a, b)))

    return math.sqrt(distance)

def multiagent_dist_between_points(a, b):
    cost = np.sum( (  np.array(a)-np.array(b) )**2 )
    return cost