"""
Created on Apr 25, 2022

@author: Wolf De Wulf, Dieter Vandesande
"""

from ctypes import alignment
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
    Configs.log("INPUT: " + str(input))
    Configs.log("INPUT LENGTH: " + str(len(input)))
    Configs.log("ALIGNMENT STARTS: " + str(subalignment_start))
    Configs.log("ALIGNMENT STARTS LENGTH: " + str(len(subalignment_start)))

    # Run minclusters to get upperbound
    incorrect_clusters = graph.clusters
    minClustersSearch(graph)
    upper_bound = len(graph.clusters)
    graph.clusters = incorrect_clusters
    
    # Setup output
    output = intvar(1, upper_bound, shape=(input[input != 0].size,), name="output")
    Configs.log(f"CREATED {input[input != 0].size} VARIABLES")
    Configs.log("MEMORY USED: " + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

    # CP using CPMPY
    model = Model()

    # all variables of the same alignment should be ordered strictly increasing.
    for a in range(subalignment_start.size - 1):
        Configs.log("NEW SUBALIGNMENT: " + str(a))
        Configs.log("in1 = " + str(in1))
        in1 = subalignment_start[a]
        out1 = subalignment_start[a]
        Configs.log("in1 = " + str(in1))
        Configs.log("out1 = " + str(out1))
        while in1 < subalignment_start[a+1]-1:
            if input[in1] == 0:
                Configs.log("SKIPPING in1: " + str(in1))
                in1 += 1
            else:
                in2 = in1 + 1
                out2 = out1 + 1
                while in2 < subalignment_start[a+1]:
                    if input[in2] == 0:
                        Configs.log("SKIPPING in2: " + str(in2))
                        in2 += 1
                    else:
                        model += output[in1] < output[in2]
                        in1 = in2
                        out1 += 1
                        Configs.log("NEXT in1: " + str(in1))
                        break
        a += 1

    Configs.log("ADDED FIRST TYPE OF CONSTRAINTS")
    Configs.log("MEMORY USED: " + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

    # nodes having different cluster ids in the input, should also have different cluster ids in the output
    in1 = 0
    out1 = 0
    while in1 < input.size-1:
        if input[in1] == 0:
            in1 += 1
        else:
            in2 = in1 + 1
            out2 = out1 + 1
            while in2 < input.size:
                if input[in2] == 0:
                    in2 += 1
                elif input[in1] != input[in2]:
                    model += output[out1] != output[out2]
                    in2 += 1
                    out2 += 1
            in1 += 1
            out1 += 1
    
    Configs.log("REACHED SOLVING")
    Configs.log("MEMORY USED: " + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

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
