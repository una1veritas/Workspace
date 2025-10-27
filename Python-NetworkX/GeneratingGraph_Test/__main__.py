'''
Created on 2025/04/18

@author: sin
'''

import networkx as nx
import matplotlib.pyplot as plt

def find_min_hub_cover(g : nx) -> set :
    remained = g.copy()
    hcover = set()
    while len(remained.edges) > 0 :
        node_ordered = sorted(remained.nodes, reverse=True, key=lambda x: g.degree(x))
        print(node_ordered)
        break
    return hcover

if __name__ == '__main__':
    # Create a random graph
    G = nx.erdos_renyi_graph(n=20, p=0.5)  # n: number of nodes, p: probability of edge creation
    '''Random Graphs: nx.erdos_renyi_graph(n, p)
    Barabási–Albert Graphs: nx.barabasi_albert_graph(n, m)
    Small-world Networks: nx.watts_strogatz_graph(n, k, p)
    '''
    print(find_min_hub_cover(G))
    # Visualize the graph
    nx.draw(G, with_labels=True, node_color="lightblue", node_size=500, font_size=10)
    plt.show()
    
    # Save the graph to a file
    #nx.write_gml(G, "graph.gml")
    
    