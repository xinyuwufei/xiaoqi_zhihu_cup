import networkx as nx
import matplotlib.pyplot as plt
import tool
import numpy as np
from statistics import mean
from itertools import combinations

def findsubsets(S, m):
    return set(combinations(S, m))

s=findsubsets([555,333,444],2)
for ss in s:
    print(ss)
TG=nx.DiGraph()

edges=[[1,2],[1,3],[1,4],[5,4]]

TG.add_edges_from(edges)
terminal_vertices=[x for x in TG.nodes_iter() if TG.out_degree(x)==0]
source_vertices=[x for x in TG.nodes_iter() if TG.in_degree(x)==0]
isolate_vertices=[x for x in TG.nodes_iter() if TG.in_degree(x)==0 and TG.out_degree()==0]

nodes=nx.topological_sort(TG)
for n in nodes:
    print(n)

print(TG.predecessors(4))
print(TG.successors(1))

