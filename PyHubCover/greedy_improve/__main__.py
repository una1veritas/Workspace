'''
Created on 2025/10/27

@author: sin
'''

import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from collections import deque

class Graph:
    def __init__(self, nxgraph : nx) :
        self.nodes = set(nxgraph.nodes)
        self.adjnodes = dict()
        for e in nxgraph.edges:
            if e[0] not in self.adjnodes :
                self.adjnodes[e[0]] = set()
            self.adjnodes[e[0]].add(e[1])
            if e[1] not in self.adjnodes :
                self.adjnodes[e[1]] = set()
            self.adjnodes[e[1]].add(e[0])
    
    def __str__(self):
        outstr = "Graph("
        outstr += str(self.nodes)
        outstr += ", "
        outstr += str(self.edge_set())
        outstr += ") "
        return outstr
    
    def edge_set(self):
        edge_pairs = set()
        for node in self.nodes:
            if node not in self.adjnodes :
                continue
            for adjnode in self.adjnodes[node]:
                if node < adjnode :
                    edge_pairs.add( (node, adjnode) )
        return edge_pairs
    
    def degree(self, node):
        return len(self.adjnodes[node])
    
    def adjacents(self, node):
        return self.adjnodes[node]
    
    def in_triangle(self, a_node, an_edge):
        if a_node in an_edge:
            return True
        return (a_node in self.adjnodes[an_edge[0]]) and (a_node in self.adjnodes[an_edge[1]])
       
def find_min_hub_cover(g : Graph) -> set :
    remained_edges = g.edge_set()
    remained_nodes = g.nodes
    hcover = set()
    while len(remained_edges) > 0 :
        covercounts = sorted([ (v, len([ e for e in remained_edges if g.in_triangle(v, e)])) for v in remained_nodes], 
                             reverse=True, key=lambda x: x[1] )
        print(covercounts)
        v = covercounts[0][0]
        
        remained_edges = set([e for e in remained_edges if not g.in_triangle(v, e)])
        hcover.add(v)
        remained_nodes.remove(v)
    return hcover
        
if __name__ == '__main__':
    # Create a random graph
    G = nx.erdos_renyi_graph(n=20, p=0.2)  # n: number of nodes, p: probability of edge creation
    '''Random Graphs: nx.erdos_renyi_graph(n, p)
    Barabási–Albert Graphs: nx.barabasi_albert_graph(n, m)
    Small-world Networks: nx.watts_strogatz_graph(n, k, p)
    '''
    # Visualize the graph
    # nx.draw(G, with_labels=True, node_color="lightblue", node_size=500, font_size=10)
    # plt.show()    
    # Save the graph to a file
    #nx.write_gml(G, "graph.gml")
    
    graph = Graph(G)
    print(graph)
    
    hcover = find_min_hub_cover(graph)
    print(hcover)
 
    cnt = 0
    dq = deque()
    while True :
        if dq :
            if len(dq) & 1 == 0 :
                node = list(hcover - set(dq))[0]
                dq.append(node)
            else:
                node = list(graph.nodes - hcover - set(dq))[0]
                dq.append(node)
        else:
            node = list(hcover)[0]
            dq.append(node)
        
        if len(dq) < 5 :
            print(dq)
        else:
            print(dq)
            last = dq.pop()
            if len(dq) & 1 == 0 :
                lst = list(hcover - set(dq))
                print(dq, lst)
                ix = lst.index(last)
                if ix + 1 == len(lst):
                    break
                dq.append(lst[ix + 1])
            else:
                lst =  list(graph.nodes - hcover - set(dq))[0]
                print(dq, lst)
                ix = lst.index(last)
                if ix + 1 == len(lst):
                    break
                dq.append(lst[ix + 1])
                