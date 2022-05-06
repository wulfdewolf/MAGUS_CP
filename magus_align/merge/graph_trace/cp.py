"""
Created on Apr 25, 2022

@author: Wolf De Wulf, Dieter Vandesande
"""

from magus_configuration import Configs
import numpy as np
from cpmpy import *
import resource
from magus_align.merge.graph_trace.min_clusters import minClustersSearch

# from magus_tools import external_tools

"""
Resolve clusters into a trace by breaking conflicting clusters apart.
We use constraint programming (MiniZinc) to search for the solution with the smallest number of clusters broken.
"""


def CPSearch(graph):
    Configs.log("Finding graph trace with constraint programming..")

    # Create cluster id matrix
    input = [0] * graph.matrixSize
    for cluster_id, cluster in enumerate(graph.clusters):
        for node in cluster:
            input[node] = cluster_id + 1

    # Transform into numpy array
    input = np.array(input)
    subalignment_start = np.array(
        [idx + 1 for idx in graph.subsetMatrixIdx] + [len(graph.matrix)]
    )

    # Run minclusters to get upperbound
    incorrect_clusters = graph.clusters
    minClustersSearch(graph)
    upper_bound = len(graph.clusters)
    graph.clusters = incorrect_clusters
    
    # Setup output
    output = intvar(1, upper_bound, shape=(input[input != 0].size,), name="output")
    print(f"CREATED {input[input != 0].size} VARIABLES")
    print("MEMORY USED: " + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

    # CP using CPMPY
    model = Model()

    # all variables of the same alignment should be ordered strictly increasing.
    i = 0
    for a in range(subalignment_start.size - 1):
        for n in range(subalignment_start[a], subalignment_start[a + 1]):
            if input[n] == 0:
                continue
            else:
                for n1 in range(n + 1, subalignment_start[a + 1] + 1):
                    if input[n1] == 0:
                        continue
                    elif n1 == subalignment_start[a + 1]:
                        break
                    else:
                        model += output[i] < output[i + 1]
                        break
                i += 1

    print("ADDED FIRST TYPE OF CONSTRAINTS")
    print("MEMORY USED: " + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

    # nodes having different cluster ids in the input, should also have different cluster ids in the output
    i = 0
    j = 0
    for n1 in range(input.size - 1):
        if input[n1] == 0:
            continue
        else:
            j = i + 1
            for n2 in range(n1 + 1, input.size):
                if input[n2] == 0:
                    continue
                elif input[n1] != input[n2]:
                    model += output[i] != output[j]
                j += 1

        i += 1

    print("REACHED SOLVING")
    print("MEMORY USED: " + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

    # Check following two possibilities for performance:
    #model.minimize(
    #    max(
    #        [
    #            output[subalignment_start[a] - 1]
    #            for a in range(1, subalignment_start.size)
    #        ]
    #    )
    #)
    # model.minimize(max(output))

    if model.solve():

        # Project output onto input
        i = 0
        for n in range(input.size):
            if input[n] != 0:
                input[n] = output[i].value()
                i += 1
    else:
        print("No solution found")

    # Transform input to clusters
    n_clusters = max(input)
    graph.clusters = [[] for _ in range(n_clusters)]
    for i in range(len(input)):
        if input[i] != 0:
            graph.clusters[input[i] - 1] = graph.clusters[input[i] - 1] + [i]

    # Remove singletons
    graph.clusters = [cluster for cluster in graph.clusters if len(cluster) > 1]
