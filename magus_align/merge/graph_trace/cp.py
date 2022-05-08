"""
Created on Apr 25, 2022

@author: Wolf De Wulf, Dieter Vandesande
"""

import copy
from magus_configuration import Configs
from cpmpy import *
from magus_align.merge.graph_trace.min_clusters import minClustersSearch
from magus_tools import external_tools

"""
Resolve clusters into a trace by breaking conflicting clusters apart.
We use constraint programming (MiniZinc) to search for the solution with the smallest number of clusters broken.
"""


def CPSearch(graph):
    Configs.log("Finding graph trace with constraint programming..")

    # Run minclusters to get upperbound
    incorrect_clusters = copy.deepcopy(graph.clusters)
    minClustersSearch(graph)
    upper_bound = len(graph.clusters)
    graph.clusters = incorrect_clusters
    
    # Run minizinc
    task, input = external_tools.runMinizincTrace(graph, upper_bound)
    task.run()

    # Decode output using input
    graph.readClustersFromCPFile(graph.tracePath, input)