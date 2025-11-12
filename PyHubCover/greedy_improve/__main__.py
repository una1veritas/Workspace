'''
Created on 2025/10/27

@author: sin
'''

import networkx as nx
import matplotlib.pyplot as plt
import itertools 
from collections import deque
import time
import random
from itertools import combinations

class UndirectedGraph:
    
    class Edge:
        def __init__(self, u, v):
            try:
                if u == v :
                    raise ValueError('an Edge of simple graph should not the self-loop.')
                if u < v :
                    self.nodes = (u, v)
                else:
                    self.nodes = (v, u)
            except:
                raise ValueError('nodes of an edge must be comparable.')
        
        def __eq__(self, another):
            return isinstance(another, self) and (self.nodes == another.nodes)
        
        def __ne__(self, another):
            return self.__eq__(another)
        
        def __str__(self):
            return f'({self.nodes[0]}, {self.nodes[1]})'
        
        def __repr__(self):
            return f'Edge({self.nodes[0]}, {self.nodes[1]})'

        def __hash__(self):
            return (hash(self.nodes[0]) <<16 + hash(self.nodes[0])) ^ (hash(self.nodes[1] <<16) + hash(self.nodes[1])) 
        
        def __contains__(self, node):
            return node in self.nodes
        
        def __getiten(self, i):
            return self.nodes[i]
        
        def __len__(self):
            return 2
                
    def __init__(self, nodes = None, edges = None, random_graph = False, node_size = None, probability = None, degree_bound = None) :
        self.nodes = set(nodes) if nodes else set()
        self.edges = set()
        self.adjnodes = dict()
        if random_graph == False:
            if edges :
                for ea in edges :
                    self.edges.add(self.Edge(ea[0], ea[1]))
            for u, v in edges:
                if u not in self.adjnodes :
                    self.adjnodes[u] = set()
                self.adjnodes[u].add(v)
                if v not in self.adjnodes :
                    self.adjnodes[v] = set()
                self.adjnodes[v].add(u)
        else:
        #def random_graph(self, size, p = 0.5, degree_bound = None):
            self.clear()
            self.nodes = set(range(0, node_size))
            degree_bound = len(self.nodes) if degree_bound == None else int(degree_bound)
            #print('degreebound = ', degree_bound)
            pairs = set([(u, v) for (u, v) in itertools.combinations(self.nodes, 2)])
            while len(pairs) :
                pair_list = list(pairs)
                (u, v) = random.choice(pair_list)
                if random.random() <= probability :
                    if (u not in self.adjnodes or len(self.adjnodes[u]) < degree_bound) and \
                    ( v not in self.adjnodes or len(self.adjnodes[v]) < degree_bound) :
                        self.add_edge(u, v)
                pairs.remove( (u, v) )

    
    def clear(self):
        self.nodes.clear()
        self.edges.clear()
        self.adjnodes.clear()
        
    def add_edge(self, u, v):
        if u in self.nodes and v in self.nodes :
            self.edges.add(self.Edge(u, v))
            if u not in self.adjnodes :
                self.adjnodes[u] = set()
            self.adjnodes[u].add(v)
            if v not in self.adjnodes :
                self.adjnodes[v] = set()
            self.adjnodes[v].add(u)
        else:
            raise ValueError('an Edge must be a pair of nodes.')
             
    def __str__(self):
        outstr = "Graph("
        outstr += str(self.nodes)
        outstr += ", "
        outstr += str(self.edges)
        outstr += ") "
        return outstr
    
    def degree(self, node):
        return len(self.adjacents(node))
    
    def adjacents(self, node):
        if node not in self.adjnodes :
            return set()
        return self.adjnodes[node]
    
    def in_triangle(self, a_node, an_edge):
        if a_node in an_edge:
            return True
        return (a_node in self.adjnodes[an_edge[0]]) and (a_node in self.adjnodes[an_edge[1]])
    
    def hub_covering_edges(self, a_node):
        edges = set()
        for adj in self.adjnodes[a_node]:
            edges.add( UndirectedGraph.Edge(a_node, adj) )
            for adjadj in self.adjnodes[adj]:
                if adjadj in self.adjnodes[a_node]:
                    edges.add( UndirectedGraph.Edge(adj, adjadj) )
        return edges
       
def find_min_hub_cover(g : UndirectedGraph) -> set :
    remained_edges = g.edges
    hcover = set()
    cover_edges = { v: set() for v in g.nodes}   # ''' keys == remained nodes, the union of all values == remained edges'''
    for (u, v) in g.edges:
        cover_edges[u].add( (u, v) )
        cover_edges[v].add( (u, v) )
        for w in g.adjnodes[u] & g.adjnodes[v] :
            cover_edges[w].add( (u, v) )
    while len(remained_edges) > 0 :
        best = 0
        node = None
        for v, covedges in cover_edges.items():
            count = len(covedges)
            if count > best :
                best = count
                node = v
        new_covered = cover_edges.pop(node)
        no_edges = list()
        for v in cover_edges.keys():
            cover_edges[v] -= new_covered
            if len(cover_edges[v]) == 0 :
                no_edges.append(v)
        for v in no_edges:
            cover_edges.pop(v)
        hcover.add(node)
        remained_edges -= new_covered
    return hcover
        
if __name__ == '__main__':
    # Create a random graph
    #G = nx.erdos_renyi_graph(n=23, p=0.2)  # n: number of nodes, p: probability of edge creation
    '''Random Graphs: nx.erdos_renyi_graph(n, p)
    Barabási–Albert Graphs: nx.barabasi_albert_graph(n, m)
    Small-world Networks: nx.watts_strogatz_graph(n, k, p)
    '''
    
    graph = UndirectedGraph(node_size = 64, random_graph=True, probability = 0.3, degree_bound = 3)
    print(f'the number of nodes = {len(graph.nodes)}, the number of edges = {len(graph.edges)} ')
    if len(graph.nodes) < 100 : print(graph)
        
    exit(0)
    
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
    