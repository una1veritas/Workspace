'''
Created on 2020/02/23

@author: Sin Shimozono
'''
from graphviz import Graph

g = Graph('Y3')
for v in [1, 2, 3, 4, 5, 6] :
    g.node(str(v))
edges = [(1, 2), (2,3), (3, 1), (4, 5), (5, 6), (6, 4), 
         (1,4), (2,5), (3, 6)]
for (u, v) in edges:
    g.edge(str(u), str(v))

g.render(view=True)
