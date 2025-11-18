'''
Created on 2025/10/27

@author: sin
'''

import networkx as nx
import matplotlib.pyplot as plt
import itertools 
import time, math, random

class UndirectedGraph:
    
    class Edge:
        def __init__(self, u, v):
            try:
                if u == v :
                    raise ValueError('an Edge of simple graph should not the self-loop.')
                self.nodes = (u, v)
            except:
                raise ValueError('nodes of an edge must be comparable.')
        
        def __eq__(self, another):
            if not isinstance(another, UndirectedGraph.Edge) :
                return False
            return self.nodes[0] in another.nodes and self.nodes[0] in another.nodes 
        
        def __ne__(self, another):
            return not self.__eq__(another)
        
        def __str__(self):
            return f'({self.nodes[0]}, {self.nodes[1]})'
        
        def __repr__(self):
            return f'Edge({self.nodes[0]}, {self.nodes[1]})'

        def __hash__(self):
            return hash(self.nodes[0]) ^ hash(self.nodes[1])
        
        def __contains__(self, node):
            return node in self.nodes
        
        def __getiten(self, i):
            return self.nodes[i]
        
        def __iter__(self):
            return self.nodes
        
        def __next__(self):
            for v in self.nodes:
                yield v
            raise StopIteration()
        
        def __len__(self):
            return 2
        
        def pair(self):
                return self.nodes
    
    def __init__(self, nodes = None, edges = None, max_degree = None, ev_ratio = None) :
        self.adjnodes = dict()
        if isinstance(nodes, int) :
            self.nodes = set(range(nodes))
        else:
            if nodes :
                self.nodes = nodes
            else:
                ValueError('Nothing for nodes specified.')
        if edges :
            if isinstance(edges, (tuple, list, set)) :
                self.edges = set()
                for ea in edges:
                    self.edges.add(self.Edge(ea))
            else:
                raise ValueError('supplied non-collection object as edges.')
        else:
            self.edges = set()
            if ev_ratio :
                ratio = float(ev_ratio)
                pairs = [pair for pair in itertools.combinations(list(self.nodes),2)]
                while len(self.edges)/len(self.nodes) < ratio :
                    edge = random.choice(pairs)
                    self.add_edge(edge[0], edge[1])
                
            else:
                if not max_degree :
                    max_degree = math.sqrt(len(self.nodes))
                ulist = list(self.nodes)
                while len(ulist) > 1:
                    u = random.choice(ulist)
                    ulist.remove(u)
                    for _ in range(max_degree):
                        v = random.choice(ulist)
                        if self.degree(u) < max_degree and self.degree(v) < max_degree :
                            if not self.adjacent(u, v) :
                                self.add_edge(u, v)
                        if self.degree(u) == max_degree :
                            break
                        if self.degree(v) == max_degree :
                            ulist.remove(v)
                            if len(ulist) == 0 :
                                break

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
    
    def __len__(self):
        return len(self.nodes)
    
    def edge_pairs(self):
        return [e.pair() for e in self.edges]
    
    def degree(self, node):
        return len(self.adjacent_nodes(node))
    
    def maximum_degree(self):
        return max([self.degree(v) for v in self.nodes])
    
    def stats(self):
        dsum = 0
        dmax = 0
        for v in self.nodes:
            d = self.degree(v)
            dmax = dmax if dmax >= d else d
            dsum += d
        return {'maximum_degree':dmax, 'average_degree': dsum/len(self)}
    
    def adjacent(self, u, v):
        return self.Edge(u,v) in self.edges
    
    def adjacent_nodes(self, node):
        if node not in self.adjnodes :
            return set()
        return self.adjnodes[node]
    
    def edges_within_triangle(self, a_node):
        if a_node not in self.nodes :
            return set([])
        edges = set()
        for adj in self.adjacent_nodes(a_node):
            edges.add(self.Edge(a_node, adj))
        for u, v in itertools.combinations(list(self.adjacent_nodes(a_node)), 2) :
            if self.adjacent(u, v) :
                edges.add(self.Edge(u,v))
        return edges
       
def find_min_hub_cover(g : UndirectedGraph, hubcover = None) -> set :
    remaining = g.edges.copy()
    covered = set()
    coverable_edges = dict()
    if hubcover :
        hubcover = set(hubcover)
        for v in hubcover:
            for e in g.edges_within_triangle(v):
                covered.add(e)
        remaining = remaining - covered
    else:
        hubcover = set()    
    for v in g.nodes - hubcover:
        coverable_edges[v] = g.edges_within_triangle(v) - covered
    
    while len(remaining) > 0 :
        best = 0
        node = None
        for v, covedges in coverable_edges.items():
            count = len(covedges)
            if count > best :
                best = count
                node = v
        if node == None : 
            break
        new_covered = coverable_edges.pop(node)
        no_edges = list()
        for v in coverable_edges.keys():
            coverable_edges[v] -= new_covered
            if len(coverable_edges[v]) == 0 :
                no_edges.append(v)
        for v in no_edges:
            coverable_edges.pop(v)
        hubcover.add(node)
        remaining -= new_covered
    return hubcover
    
def neighbor(g: UndirectedGraph, hcover, node):
    dist1 = g.adjacent_nodes(node)
    dist2 = set()
    for u in dist1:
        dist2 = dist2 | g.adjacent_nodes(u)
    dist3 = set()
    for u in dist2:
        dist3 = dist3 | g.adjacent_nodes(u)
    adjs = dist1 | dist2 | dist3
    hcover = hcover - adjs
    hcover = find_min_hub_cover(g, hcover)
    return hcover

if __name__ == '__main__':
    # Create a random graph
    #G = nx.erdos_renyi_graph(n=23, p=0.2)  # n: number of nodes, p: probability of edge creation
    '''Random Graphs: nx.erdos_renyi_graph(n, p)
    Barabási–Albert Graphs: nx.barabasi_albert_graph(n, m)
    Small-world Networks: nx.watts_strogatz_graph(n, k, p)
    '''
    
    graph = UndirectedGraph(nodes = 2000, ev_ratio = 3)
    print(f'the number of nodes = {len(graph.nodes)},\nthe number of edges = {len(graph.edges)} ')
    if len(graph.nodes) < 128 : 
        print(graph)
    print(graph.stats())

    # Visualize the graph
    if len(graph) < 1000 :
        nxg = nx.Graph()
        nxg.add_nodes_from(graph.nodes)
        nxg.add_edges_from(graph.edge_pairs())
        nx.draw(nxg, with_labels=True, node_color="lightblue", edge_color="black",node_size=100, font_size=10)
        plt.show()
        # Save the graph to a file
        #nx.write_gml(G, "graph.gml")
    
    print('starting greedy algorithm.')
    start = time.perf_counter()
    hcover = find_min_hub_cover(graph)
    end = time.perf_counter()
    print("HubCover = ", hcover, "\nsize = ", len(hcover))
    print(f"Elapsed: {end - start:.6f} seconds")

    print('\nstarting local search improvement loop.')
    size_at_start = len(hcover)
    nodelst = list(graph.nodes)
    imprv = 0
    start = time.perf_counter()
    startidx = random.randint(0, len(nodelst) - 1)
    while True:
        for idx in range(len(nodelst)):
            v = nodelst[(startidx + idx) % len(nodelst)]
            new_hcover = neighbor(graph, hcover, v)
            if len(new_hcover) < len(hcover) :
                hcover = new_hcover
                print(f'updated, size {len(hcover)}, from node {v}')
                imprv += 1
                startidx = (startidx + idx + 1) % len(nodelst)
                break
        else:
            print(f'arrived at a local optimum.')
            break
    end = time.perf_counter()
    print(f"Elapsed: {end - start:.6f} seconds")
    print("HubCover = ", hcover, "\nsize = ", len(hcover))
    print(f'Improved {imprv} times,', end='')
    rate = (1.0 - float(len(hcover))/size_at_start)*100
    print(f'{-rate:5.2}%')
        
    exit(0)
    
    '''
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
        
        # extend deqs 
        if len(deqs) < depth_limit :
            selected = [deq[0] for deq in deqs]
            if len(deqs) & 1 == 0 : 
                # adding odd-th candidates, from hcover 
                deqs.append([v for v in hcov_lst if v not in selected])
            else: 
                # adding even-th candidates, from remeined 
                deqs.append([v for v in rem_lst if v not in selected])
        
        # enumerate it 
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

        # go next or reduce deqs 
        while len(deqs[-1]) == 0 :
            deqs.pop()
            if len(deqs) == 0 :
                break
            deqs[-1].pop(0)
        
        if len(deqs) == 0 :
            break
        
        # debug 
        cnt += 1
        # if cnt > loop_lim : 
        #     break

    end = time.perf_counter()
    print(f"Elapsed: {end - start:.6f} seconds")
    print('finished?', cnt)
    '''