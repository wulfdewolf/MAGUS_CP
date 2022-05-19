"""
Created on May 9, 2022

@author: Dieter Vandesande & Wolf De Wulf
"""

import heapq
from typing import OrderedDict

from magus_configuration import Configs

"""
Resolve clusters into a trace by breaking conflicting clusters apart.
We use A* to search for the path of cluster breaks with the smallest number of clusters broken.
(REIMPLEMENTATION)
"""


class State:
    def __init__(self, g, h, frontier, broken_clusters, max_cut):
        self.frontier = frontier
        self.broken_clusters = broken_clusters
        self.g = g
        self.h = h
        self.max_cut = max_cut

    def __lt__(self, other):
        self.f() < other.f()

    def f(self):
        return self.h + self.g

    def develop(self, problem_matrix, graph, cluster_idxs, agression):

        legal = True
        while legal:

            legal = False
            self.new_broken_clusters = []
            visited = set()
            self.safe_frontier = True

            # Loop over frontier
            for suba in self.frontier:
                frontier_idx = self.frontier[suba]

                # Verify if safe frontier
                if frontier_idx <= self.max_cut[suba]:
                    self.safe_frontier = False

                # Verify if frontier has reached end for this subalignment
                if frontier_idx == len(problem_matrix[suba]):
                    continue

                # Verify if already visited
                cluster_idx, pos = problem_matrix[suba][frontier_idx]
                if (cluster_idx, suba) in visited:
                    continue

                # Get cluster
                if (cluster_idx, suba) in self.broken_clusters:
                    cluster = self.broken_clusters[cluster_idx, suba]
                else:
                    cluster = graph.clusters[cluster_idx]

                # Check if legal
                good_side, bad_side = [], []
                for node in cluster:
                    sub, pos = graph.matSubPosMap[node]
                    visited.add((cluster_idx, sub))
                    bidx = cluster_idxs[cluster_idx][sub]
                    diff = bidx - self.frontier[sub]
                    if diff == 0:
                        good_side.append(node)
                    else:
                        bad_side.append(node)

                if len(bad_side) == 0:
                    # Legal
                    for node in cluster:
                        sub, pos = graph.matSubPosMap[node]
                        self.frontier[sub] = cluster_idxs[cluster_idx][sub] + 1
                    self.g += 1
                    self.h -= 1
                    legal = True
                    break
                else:
                    # Break
                    self.new_broken_clusters.append((cluster_idx, good_side, bad_side))

        # If broken, add agression
        if not self.safe_frontier and len(self.broken_clusters) != 0:
            self.h *= agression

    def generate_neighbors(
        self,
        problem_matrix,
        graph,
        cluster_idxs,
        agression,
    ):

        neighbors = []
        for cluster_idx, good_side, bad_side in self.new_broken_clusters:

            frontier_copy = dict(self.frontier)
            broken_clusters_copy = dict(self.broken_clusters)
            max_cut_copy = dict(self.max_cut)

            for node in good_side:
                sub, pos = graph.matSubPosMap[node]
                broken_clusters_copy[cluster_idx, sub] = good_side
                max_cut_copy[sub] = max(max_cut_copy[sub], cluster_idxs[cluster_idx][sub])

            for node in bad_side:
                sub, pos = graph.matSubPosMap[node]
                broken_clusters_copy[cluster_idx, sub] = bad_side
                max_cut_copy[sub] = max(max_cut_copy[sub], cluster_idxs[cluster_idx][sub])

            neighbor = State(
                self.g, self.h + 1, frontier_copy, broken_clusters_copy, max_cut_copy
            )
            neighbor.develop(
                problem_matrix,
                graph,
                cluster_idxs,
                agression,
            )
            neighbors.append(neighbor)

        return neighbors

    def generate_clusters(self, problem_matrix, graph, cluster_idxs):

        # Start from initial frontier
        frontier = {suba: 0 for suba in problem_matrix}

        ordered_clusters = []
        legal = True
        while legal:

            legal = False

            # Loop over frontier
            for suba in frontier:
                good = True
                frontier_idx = frontier[suba]

                # Verify if frontier has reached end for this subalignment
                if frontier_idx == len(problem_matrix[suba]):
                    continue
                cluster_idx, pos = problem_matrix[suba][frontier_idx]

                # Get cluster
                if (cluster_idx, suba) in self.broken_clusters:
                    cluster = self.broken_clusters[cluster_idx, suba]
                else:
                    cluster = graph.clusters[cluster_idx]
                
                # Verify if good
                for node in cluster:
                    sub, pos = graph.matSubPosMap[node]
                    if cluster_idxs[cluster_idx][sub] != frontier[sub]:
                        good = False
                        break
                
                if good:

                    # Select cluster and move frontier
                    ordered_clusters.append(cluster)
                    for node in cluster:
                        sub, pos = graph.matSubPosMap[node]
                        frontier[sub] = cluster_idxs[cluster_idx][sub] + 1
                    legal = True
                    break

        return ordered_clusters


def reimplementedMinClustersSearch(graph):

    # Create problem matrix and reverse cluster idx map
    problem_matrix = {}
    cluster_idxs = {}
    for cluster_idx, cluster in enumerate(graph.clusters):
        for node in cluster:
            suba, pos = graph.matSubPosMap[node]
            problem_matrix[suba] = problem_matrix.get(suba, []) + [(cluster_idx, pos)]
            cluster_idxs[cluster_idx] = {}

    # Sort subalignments and create initial frontier
    initial_frontier = {}
    initial_max_cut = {}
    best_frontier = {}
    last_frontier_state = None
    for suba in problem_matrix:
        problem_matrix[suba].sort(key=lambda c: c[1])

        initial_frontier[suba] = 0
        best_frontier[suba] = -1
        initial_max_cut[suba] = -1

        # Fill reverse cluster idx map
        for i in range(len(problem_matrix[suba])):
            cluster_idx = problem_matrix[suba][i][0]
            cluster_idxs[cluster_idx][suba] = i

    # Open list
    open_list = []

    # Closed list
    closed_list = set()

    # Search parameters
    agression = 1.0
    greedy = False

    # Start state
    current_state = State(
        0,
        len(graph.clusters),
        initial_frontier,
        {},
        initial_max_cut,
    )
    current_state.develop(problem_matrix, graph, cluster_idxs, agression)

    # Add the start state
    heapq.heappush(open_list, current_state)

    # Loop until open list is empty
    while len(open_list) > 0:

        # If open list limit is reached
        heap_cleared = False
        if len(open_list) > Configs.searchHeapLimit:

            # Increase agression
            Configs.log(
                "Open list limit {} reached.. Truncating to last frontier".format(
                    Configs.searchHeapLimit
                )
            )
            if aggression == 1.0:
                aggression = 1.2
                Configs.log("Increasing aggression to {}..".format(aggression))
            elif aggression < 8:
                aggression = int(aggression) * 2
                Configs.log("Increasing aggression to {}..".format(aggression))
            else:
                Configs.log("Setting search strategy to fully greedy..")
                aggression = 1.0
                greedy = True

            # Clear
            open_list = []
            closed_list = set()

            # Go back to best frontier state
            heapq.heappush(open_list, last_frontier_state)
            heap_cleared = True

        # Get the current state
        current_state = heapq.heappop(open_list)

        if len(current_state.new_broken_clusters) == 0:

            # If no breaks, ordering is done
            graph.clusters = current_state.generate_clusters(
                problem_matrix, graph, cluster_idxs
            )
            return

        else:

            # Verify if state has been visited yet
            state_hash = tuple(
                [current_state.frontier[suba] for suba in problem_matrix]
            )
            if state_hash in closed_list:
                continue
            else:
                closed_list.add(state_hash)

            # Verify if new sub frontier
            new_sub_frontier = True
            for suba in current_state.frontier:
                if current_state.frontier[suba] <= best_frontier[suba]:
                    new_sub_frontier = False
                    break

            if new_sub_frontier:
                Configs.log("Reached new best frontier")
                best_frontier = current_state.frontier
                last_frontier_state = current_state
                greedy = False

            # Verify if new safe frontier
            if current_state.safe_frontier and not heap_cleared:
                Configs.log(
                    "Safe frontier reached.. dumping {} from open list and resetting aggression..".format(
                        len(open_list)
                    )
                )
                last_frontier_state = current_state
                open_list.clear()
                closed_list = set()
                aggression = 1.0
                greedy = False

            # Generate neighbors
            neighbors = current_state.generate_neighbors(
                problem_matrix, graph, cluster_idxs, agression
            )

            # Add to open list
            if greedy:
                neighbor = min(neighbors, key=lambda neighbor : neighbor.f())
                heapq.heappush(open_list, neighbor)
            else:
                for neighbor in neighbors:
                    heapq.heappush(open_list, neighbor)