'''
Created on May 9, 2022

@author: Dieter Vandesande & Wolf De Wulf
'''

import heapq
import random

from magus_configuration import Configs

'''
Resolve clusters into a trace by breaking conflicting clusters apart.
We use A* to search for the path of cluster breaks with the smallest number of clusters broken.
(REIMPLEMENTATION)
'''

# A-star template from https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2

class State():

    def __init__(self, unordered, ordered):
        self.unordered = unordered
        self.ordered = ordered

    def __eq__(self, other):
        return self.unordered == other.unordered and self.ordered == other.ordered

    def __lt__(self, other):
        return self.f() < other.f()

    def __hash__(self):
        return hash(id(self))

    def is_ordered(self):
        len(self.unordered) == 0

    def f(self):
        return self.h() + self.g()

    def h(self):
        return len(self.unordered)

    def g(self):
        return len(self.ordered)


def reimplementedMinClustersSearch(graph):

    # Create start and end node
    start_node = State(graph.clusters, [])

    # Initialize both open and closed list
    open_list = []
    closed_list = set()

    # Add the start node
    heapq.heappush(open_list, start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_state = heapq.heappop(open_list)
        Configs.log(f"New state reached: unordered length = {len(current_state.unordered)}, ordered_length = {len(current_state.ordered)}")

        # Pop current off open list, add to closed list
        closed_list.add(current_state)

        # Found the goal
        if current_state.is_ordered():
            graph.clusters = current_state.ordered
            return 

        # Generate neighbors
        neighbors = generate_neighbors(graph, current_state)

        # Loop through neighbors
        for neighbor in neighbors:

            # Neighbor is in the closed list
            if neighbor in closed_list:
                continue

            # Neighbor is already in the open list
            if neighbor in open_list:
                continue

            # Add the child to the open list
            heapq.heappush(open_list, neighbor)


def generate_neighbors(graph, state):

    neighbors = []
    for cluster_idx, cluster in enumerate(state.unordered):

        # Check if cluster can be added to ordered
        legal = True
        for node in cluster:
            node_subalignment, node_position = graph.matSubPosMap[node]

            for ordered_cluster in state.ordered:
                for ordered_node in ordered_cluster:
                    ordered_node_subalignment, ordered_node_position = graph.matSubPosMap[ordered_node] 

                    if node_subalignment == ordered_node_subalignment:
                        legal = legal and node_position < ordered_node_position

        # If so, add the transitioned node as neighbor
        if legal:
            neighbors.append(State(state.unordered[:cluster_idx] + state.unordered[cluster_idx+1:], state.ordered + [cluster]))
    if len(neighbors) != 0:
        Configs.log(f"Found legal {len(neighbors)} moves!")

    # If no legal moves
    if len(neighbors) == 0:

        # Pick random unordered cluster to break apart
        random_cluster_idx = random.randrange(len(state.unordered))
        random_cluster = state.unordered[random_cluster_idx]

        # Split it at a random index, discard singleton clusters
        random_split_idx = random.randrange(len(random_cluster))
        if len(random_cluster) == 2:
            split_unordered = state.unordered[:random_cluster_idx] + state.unordered[random_cluster_idx+1:]
        elif len(random_cluster[:random_split_idx]) == 1:
            split_unordered = state.unordered[:random_cluster_idx] + state.unordered[random_cluster_idx+1:] + random_cluster[random_split_idx:]
        elif len(random_cluster[random_split_idx:]) == 1:
            split_unordered = state.unordered[:random_cluster_idx] + state.unordered[random_cluster_idx+1:] + random_cluster[:random_split_idx]
        else:
            split_unordered = state.unordered[:random_cluster_idx] + state.unordered[random_cluster_idx+1:] + random_cluster[:random_split_idx] + random_cluster[random_split_idx:]

        # Add transitioned node as neighbor
        neighbors.append(State(split_unordered, state.ordered[:]))
        Configs.log("Had to break a cluster!")

    return neighbors