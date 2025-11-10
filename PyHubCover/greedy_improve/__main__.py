'''
Created on 2025/10/27

@author: sin
'''

import networkx as nx
import matplotlib.pyplot as plt
# from itertools import combinations
from collections import deque
import time

class Graph:
    def __init__(self, nxgraph : nx) :
        self.nodes = set(nxgraph.nodes)
        self.edges = set()
        self.adjdict = dict()
        for (u, v) in nxgraph.edges:
            if u < v :
                self.edges.add( (u, v) )
            else:
                self.edges.add( (v, u) )
            if u not in self.adjdict :
                self.adjdict[u] = set()
            self.adjdict[u].add(v)
            if v not in self.adjdict :
                self.adjdict[v] = set()
            self.adjdict[v].add(u)
    
    def __str__(self):
        outstr = "Graph("
        outstr += str(self.nodes)
        outstr += ", "
        outstr += str(self.edges)
        outstr += ") "
        return outstr
    
    def degree(self, node):
        return len(self.adjdict[node])
    
    def adjacent_node_set(self, node):
        return self.adjdict[node]
    
    def adjacent(self, u, v):
        return v in self.adjdict[u]
    
    '''
    def in_triangle(self, a_node, an_edge):
        if a_node in an_edge:
            return True
        return (a_node in self.adjdict[an_edge[0]]) and (a_node in self.adjdict[an_edge[1]])
    
    def edges_within_triangles(self, a_node):
        in_triangles = set()
        for adj in self.adjdict[a_node]:
            if a_node < adj :
                in_triangles.add( (a_node, adj) )
            else:
                in_triangles.add( (adj, a_node) )
            for adjadj in self.adjdict[adj]:
                if adjadj in self.adjdict[a_node]:
                    if adj < adjadj :
                        in_triangles.add( (adj, adjadj) )
                    else:
                        in_triangles.add( (adjadj, adj) )
        return in_triangles
    '''
    
def find_min_hub_cover(g : Graph) -> set :
    remained_edges = g.edges.copy()
    hcover = set()
    coverables = dict()
    for (u, v) in g.edges :
        if u not in coverables: coverables[u] = set()
        if v not in coverables: coverables[v] = set()
        coverables[u].add( (u, v) )
        coverables[v].add( (u, v) )
        for w in g.adjdict[u] & g.adjdict[v] :
            if w not in coverables: coverables[w] = set()
            coverables[w].add( (u, v) )
    while len(remained_edges) > 0 :
        best = 0
        node = None
        for v, edges in coverables.items():
            count = len(edges)
            if count > best :
                best = count
                node = v
        new_covered = coverables.pop(node)
        no_edges = list()
        for v in coverables.keys():
            coverables[v] -= new_covered
            if len(coverables[v]) == 0 :
                no_edges.append(v)
        for v in no_edges:
            coverables.pop(v)
        hcover.add(node)
        remained_edges -= new_covered
    return hcover
        
if __name__ == '__main__':
    # Create a random graph
    G = nx.erdos_renyi_graph(n=1023, p=0.7)  # n: number of nodes, p: probability of edge creation
    '''Random Graphs: nx.erdos_renyi_graph(n, p)
    Barabási–Albert Graphs: nx.barabasi_albert_graph(n, m)
    Small-world Networks: nx.watts_strogatz_graph(n, k, p)
    '''
    
    graph = Graph(G)
    print(f'the number of nodes = {len(graph.nodes)}, the number of edges = {len(graph.edges)} ')
    
    start = time.perf_counter()
    hcover = find_min_hub_cover(graph)
    end = time.perf_counter()
    print("HubCover = ", hcover, "\nsize = ", len(hcover))
    print(f"Elapsed: {end - start:.6f} seconds")
    remained = graph.nodes - hcover
    print()

    # Visualize the graph
    nx.draw(G, with_labels=True, node_color="lightblue", node_size=500, font_size=10)
    plt.show()    
    # Save the graph to a file
    #nx.write_gml(G, "graph.gml")
    
    exit(0)
    
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
            # if cnt & 0x7fffff == 0 :
            #     print(cnt>>10, deqs)
            #     print(selected)
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
    