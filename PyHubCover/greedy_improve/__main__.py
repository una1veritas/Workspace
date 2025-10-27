'''
Created on 2025/10/27

@author: sin
'''

import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations


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
        outstr += str(self.edges())
        outstr += ") "
        return outstr
    
    def edges(self):
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
    
       
def find_min_hub_cover(g : Graph) -> set :
    remained_edges = g.edges()
    remained_nodes = g.nodes
    hcover = set()
    while len(remained_edges) > 0 :
        covercounts = sorted([ (v, len([ e for e in remained_edges if v in e])) for v in remained_nodes], 
                             reverse=True, key=lambda pair: pair[1] )
        print(covercounts)
        v = covercounts[0][0]
        for adj in g.adjacents(v) :
            if (v, adj) in remained_edges :
                remained_edges.remove( (v, adj) )
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
    nx.draw(G, with_labels=True, node_color="lightblue", node_size=500, font_size=10)
    plt.show()    
    # Save the graph to a file
    #nx.write_gml(G, "graph.gml")
    
    graph = Graph(G)
    print(graph)
    
    res = find_min_hub_cover(graph)
    print(res)
    
    for t in combinations(graph.nodes, 3):
        print(t)