
'''
Created on Apr 25, 2022

@author: Wolf De Wulf, Dieter Vandesande
'''

from magus_configuration import Configs
from magus_tools import external_tools

'''
Resolve clusters into a trace by breaking conflicting clusters apart.
We use constraint programming (MiniZinc) to search for the solution with the smallest number of clusters broken.
'''

def CPSearch(graph):
    Configs.log("Finding graph trace with constraint programming..")
    external_tools.runMinizincTrace(graph.clusters, graph.matSubPosMap, graph.workingDir, graph.tracePath).run()
    graph.readClustersFromFile(graph.tracePath)