"""
Created on Apr 25, 2022

@author: Wolf De Wulf, Dieter Vandesande
"""

import copy
from magus_configuration import Configs
from magus_align.merge.graph_trace.min_clusters import minClustersSearch
from magus_align.merge.graph_cluster.clean_clusters import purgeClusterViolations
from magus_tools import external_tools

"""
Resolve clusters into a trace by breaking conflicting clusters apart.
We use constraint programming (MiniZinc) to search for the solution with the smallest number of clusters broken.
"""

def CPSearch(graph):
    Configs.log("Finding graph trace with constraint programming..")

    # Run minclusters to get upperbound
    incorrect_clusters = copy.deepcopy(graph.clusters)
    print(f"len graph clusters before violations removed =  {len(graph.clusters)}" )
    nr_row_violations, nr_col_violations = purgeClusterViolations(graph)
    print(f"len graph clusters after violations removed = {len(graph.clusters)}" )
    minClustersSearch(graph)
    print(f"row violations = {nr_row_violations} , nr_col_violations = {nr_col_violations}")
    upper_bound = len(graph.clusters) + nr_row_violations + nr_col_violations
    Configs.log(f"nr of clusters found by minClusters = {upper_bound}")
    graph.clusters = incorrect_clusters
        
    # Run minizinc
    task, input = external_tools.runMinizincTrace(graph, upper_bound)
    task.run()

    # Decode output using input
    graph.readClustersFromCPFile(graph.tracePath, input)
    Configs.log(f"nr of clusters found by CP = {len(graph.clusters)}")
