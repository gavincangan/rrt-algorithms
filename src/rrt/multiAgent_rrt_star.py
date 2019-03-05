# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
import random
import numpy as np
from operator import itemgetter

from src.utilities.geometry import multiagent_dist_between_points

from src.rrt.tree import Tree

from src.rrt.heuristics import cost_to_go
# from src.rrt.heuristics import segment_cost, path_cost
from src.rrt.heuristics import multiagent_segment_cost, multiagent_path_cost

from src.rrt.rrt import RRT

class MultiAgentRRTStar(RRT):
    def __init__(self, X, Q, x_init, x_goal, max_samples, r, prc=0.01, rewire_count=None):
        """
        RRT* Search
        :param X: Search Space
        :param Q: list of lengths of edges added to tree
        :param x_init: tuple of tuples, initial locations
        :param x_goal: tuple of tuples, goal locations
        :param max_samples: max number of samples to take
        :param r: resolution of points to sample along edge when checking for collisions
        :param prc: probability of checking whether there is a solution
        :param rewire_count: number of nearby vertices to rewire
        """
        self.num_agents = int( len(x_init) / X.dimensions )
        super().__init__(X, Q, x_init, x_goal, max_samples, r, prc)
        self.rewire_count = rewire_count if rewire_count is not None else 0
        self.c_best = float('inf')  # length of best solution thus far

    def get_nearby_vertices(self, tree, x_init, x_new):
        """
        Get nearby vertices to new vertex and their associated costs, number defined by rewire count
        :param tree: tree in which to search
        :param x_init: starting vertex used to calculate path cost
        :param x_new: vertex around which to find nearby vertices
        :return: list of nearby vertices and their costs, sorted in ascending order by cost
        """
        X_near = self.nearby(tree, x_new, self.current_rewire_count(tree))
        L_near = [(x_near, multiagent_path_cost(self.trees[tree].E, x_init, x_near) + multiagent_segment_cost(x_near, x_new)) for
                  x_near in X_near]
        # noinspection PyTypeChecker
        L_near.sort(key=itemgetter(0))

        return L_near

    def rewire(self, tree, x_new, L_near):
        """
        Rewire tree to shorten edges if possible
        Only rewires vertices according to rewire count
        :param tree: int, tree to rewire
        :param x_new: tuple, newly added vertex
        :param L_near: list of nearby vertices used to rewire
        :return:
        """
        for x_near, c_near in L_near:
            curr_cost = multiagent_path_cost(self.trees[tree].E, self.x_init, x_near)
            tent_cost = multiagent_path_cost(self.trees[tree].E, self.x_init, x_new) + c_near

            if tent_cost < curr_cost: # and self.X.collision_free(x_near, x_new, self.r):
                self.trees[tree].E[x_near] = x_new

    def connect_shortest_valid(self, tree, x_new, L_near):
        """
        Connect to nearest vertex that has an unobstructed path
        :param tree: int, tree being added to
        :param x_new: tuple, vertex being added
        :param L_near: list of nearby vertices
        """
        # check nearby vertices for total cost and connect shortest valid edge
        for x_near, c_near in L_near:
            if c_near + cost_to_go(x_near, self.x_goal) < self.c_best and self.connect_to_point(tree, x_near, x_new):
                break

    def current_rewire_count(self, tree):
        """
        Return rewire count
        :param tree: tree being rewired
        :return: rewire count
        """
        # if no rewire count specified, set rewire count to be all vertices
        if self.rewire_count is not None:
            return self.trees[tree].V_count

        # max valid rewire count
        return min(self.trees[tree].V_count, self.rewire_count)

    def rrt_star(self):
        """
        Based on algorithm found in: Incremental Sampling-based Algorithms for Optimal Motion Planning
        http://roboticsproceedings.org/rss06/p34.pdf
        :return: set of Vertices; Edges in form: vertex: [neighbor_1, neighbor_2, ...]
        """

        self.add_vertex(0, self.x_init)
        self.add_edge(0, self.x_init, None)

        while True:
            for q in self.Q:  # iterate over different edge lengths
                for i in range(q[1]):  # iterate over number of edges of given length to add
                    x_new, x_nearest = self.new_and_near(0, q)
                    if x_new is None:
                        continue

                    # get nearby vertices and cost-to-come
                    L_near = self.get_nearby_vertices(0, self.x_init, x_new)

                    # check nearby vertices for total cost and connect shortest valid edge
                    self.connect_shortest_valid(0, x_new, L_near)

                    if x_new in self.trees[0].E:
                        # rewire tree
                        self.rewire(0, x_new, L_near)

                    # probabilistically check if solution found
                    if self.prc and random.random() < self.prc:
                        print("Checking if can connect to goal at", str(self.samples_taken), "samples")
                        path = self.get_path()
                        if path is not None:
                            return path

                    # check if can connect to goal after generating max_samples
                    if self.samples_taken >= self.max_samples:
                        return self.get_path()

    def new_and_near(self, tree, q):
        """
        Return a new steered vertex and the vertex in tree that is nearest
        :param tree: int, tree being searched
        :param q: length of edge when steering
        :return: vertex, new steered vertex, vertex, nearest vertex in tree to new vertex
        """
        x_rand = tuple(np.hstack([ self.X.sample_free() for _ in range(self.num_agents) ]))
        x_nearest = self.get_nearest(tree, x_rand)
        x_new = self.steer(x_nearest, x_rand, q[0])

        # check if new point is in X_free and not already in V
        if not self.trees[0].V.count(x_new) == 0: # or not self.X.obstacle_free(x_new):
            return None, None

        self.samples_taken += 1

        return x_new, x_nearest

    def can_connect_to_goal(self, tree):
        """
        Check if the goal can be connected to the graph
        :param tree: rtree of all Vertices
        :return: True if can be added, False otherwise
        """
        x_nearest = self.get_nearest(tree, self.x_goal)
        if self.x_goal in self.trees[tree].E and x_nearest in self.trees[tree].E[self.x_goal]:
            # tree is already connected to goal using nearest vertex
            return True

        # if self.X.collision_free(x_nearest, self.x_goal, self.r):  # check if obstacle-free
        if multiagent_dist_between_points( x_nearest, self.x_goal ) < 5000:
            return True

        return False

    def connect_to_point(self, tree, x_a, x_b):
        """
        Connect vertex x_a in tree to vertex x_b
        :param tree: int, tree to which to add edge
        :param x_a: tuple, vertex
        :param x_b: tuple, vertex
        :return: bool, True if able to add edge, False if prohibited by an obstacle
        """
        if self.trees[tree].V.count(x_b) == 0: # and self.X.collision_free(x_a, x_b, self.r):
            self.add_vertex(0, x_b)
            self.add_edge(0, x_b, x_a)

            return True

        return False

    def add_tree(self):
        """
        Create an empty tree and add to trees
        """
        self.trees.append( Tree(self.X, dimensions=(self.X.dimensions * self.num_agents) ) )

    def steer(self, start, goal, distance):
        """
        Return a point in the direction of the goal, that is distance away from start
        :param start: start location
        :param goal: goal location
        :param distance: distance away from start
        :return: point in the direction of the goal, distance away from start
        """
        ab = np.empty(len(start), np.float)  # difference between start and goal
        for i, (start_i, goal_i) in enumerate(zip(start, goal)):
            ab[i] = goal_i - start_i

        ab = tuple(ab)
        zero_vector = tuple(np.zeros(len(ab)))

        ba_length = multiagent_dist_between_points(zero_vector, ab)  # get length of vector ab
        unit_vector = np.fromiter((i / ba_length for i in ab), np.float, len(ab))
        # scale vector to desired length
        scaled_vector = np.fromiter((i * distance for i in unit_vector), np.float, len(unit_vector))
        steered_point = np.add(start, scaled_vector)  # add scaled vector to starting location for final point

        # if point is out-of-bounds, set to bound
        for dim, dim_range in enumerate(self.X.dimension_lengths):
            if steered_point[dim] < dim_range[0]:
                steered_point[dim] = dim_range[0]
            elif steered_point[dim] > dim_range[1]:
                steered_point[dim] = dim_range[1]

        return tuple(steered_point)


    # @staticmethod
    # def multiagent_path_cost( E, a, b ):
    #     """
    #     Cost of the unique path from x_init to x
    #     :param E: edges, in form of E[child] = parent
    #     :param a: initial location
    #     :param b: goal location
    #     :return: segment_cost of unique path from x_init to x
    #     """
    #     cost = 0
    #     while not b == a:
    #         p = E[b]
    #         cost += multiagent_dist_between_points(b, p)
    #         b = p

    #     return cost

    # @staticmethod
    # def multiagent_segment_cost(x_near, x_new):
    #     return multiagent_dist_between_points(x_near, x_new)