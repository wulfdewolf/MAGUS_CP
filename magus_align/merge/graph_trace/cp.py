
'''
Created on Apr 25, 2022

@author: Wolf De Wulf, Dieter Vandesande
'''

from magus_configuration import Configs
import numpy as np
from cpmpy import *
#from magus_tools import external_tools

'''
Resolve clusters into a trace by breaking conflicting clusters apart.
We use constraint programming (MiniZinc) to search for the solution with the smallest number of clusters broken.
'''

def CPSearch(graph):
    Configs.log("Finding graph trace with constraint programming..")
    #external_tools.runMinizincTrace(graph, graph.workingDir, graph.tracePath).run()
    #graph.readClustersFromCPFile(graph.tracePath)

    # Create cluster id matrix
    input = [0] * graph.matrixSize
    for cluster_id, cluster in enumerate(graph.clusters):
        for node in cluster:
            input[node] = cluster_id + 1

    # Fill singleton clusters
    new_singleton_id = len(graph.clusters)
    for node in range(len(input)):
        if input[node] == 0:
            input[node] = new_singleton_id
            new_singleton_id += 1

    # Transform into numpy array
    input = np.array(input)
    subalignment_start = np.array([idx + 1 for idx in graph.subsetMatrixIdx] + [len(graph.matrix)])
    
    # Setup output 
    output = intvar(1,input.size,  shape=input.shape, name="output")
    
    # CP using CPMPY
    model = Model()
    
    for a in range(subalignment_start.size - 1):
        for n in range(subalignment_start[a], subalignment_start[a+1]-1):
            model += (output[n] < output[n+1])
    
    
    for n1 in range(input.size - 1):
        for n2 in range(n1+1, input.size):
            print(str(n1) + " = " + str(input[n1]) +  " , " + str(n2) + " = " + str(input[n2])  )
            if input[n1] != input[n2]:
                model += output[n1] != output[n2]
    
    # Check following two possibilities for performance:
    model.minimize(max([output[subalignment_start[a]-1] for a in range(1, subalignment_start.size)]))
    # model.minimize(max(output))
    
    print(model)
    
    if model.solve():
        print(output.value())
    else:
        print("No solution found")

    # Transform output to clusters
    n_clusters = max(output) 
    graph.clusters = [[] for _ in range(n_clusters)]

    # Loop over matrix
    for i in range(len(output)):
        graph.clusters[output[i]-1] = graph.clusters[output[i]-1] + [i]

    # Remove singletons
    graph.clusters = [cluster for cluster in graph.clusters if len(cluster) > 1]
