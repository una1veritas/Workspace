'''
Created on 2025/10/27

@author: sin
'''

import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from collections import deque
import time

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
    G = nx.erdos_renyi_graph(n=250, p=0.2)  # n: number of nodes, p: probability of edge creation
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
    remained = graph.nodes - hcover
    print()
    
    cnt = 0
    loop_lim = float('inf') # 20000
    deqs = deque()
    depth_limit = 5
    hcov_lst = sorted(list(hcover)) # make sorted for debug
    rem_lst = sorted(list(remained))
    print("hcov = ", hcov_lst, "\nrem = ", rem_lst)
    last = None
    deqs.append(hcov_lst.copy())
    
    #lastone = []
    start = time.perf_counter()
    while True :
        
        ''' extend deqs '''
        if len(deqs) < depth_limit :
            selected = [deq[0] for deq in deqs]
            if len(deqs) & 1 == 0 : 
                ''' adding odd-th candidates, from hcover '''
                deqs.append([v for v in hcov_lst if v not in selected])
            else: 
                ''' adding even-th candidates, from remeined '''
                deqs.append([v for v in rem_lst if v not in selected])
        
        ''' enumerate it '''
        if len(deqs) == depth_limit :
            selected = [deq[0] for deq in deqs]
            if cnt & 0xfffff == 0 :
                print(cnt>>10, deqs)            
                #print(selected)
            # if lastone :
            #     if not (lastone < selected) :
            #         raise ValueError('something going wrong!') 
            # lastone = selected
            deqs[-1].pop(0)

        ''' go next or reduce deqs '''
        while len(deqs[-1]) == 0 :
            deqs.pop()
            if len(deqs) == 0 :
                break
            deqs[-1].pop(0)
        
        if len(deqs) == 0 :
            break
        
        ''' debug '''
        cnt += 1
        # if cnt > loop_lim : 
        #     break

    end = time.perf_counter()
    print(f"Elapsed: {end - start:.6f} seconds")
    print('finished?', cnt)
    