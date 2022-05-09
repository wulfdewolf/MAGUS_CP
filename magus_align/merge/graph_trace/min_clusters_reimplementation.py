'''
Created on May 9, 2022

@author: Dieter Vandesande & Wolf De Wulf
'''

import heapq

from magus_configuration import Configs

'''
Resolve clusters into a trace by breaking conflicting clusters apart.
We use A* to search for the path of cluster breaks with the smallest number of clusters broken.
(REIMPLEMENTATION)
'''