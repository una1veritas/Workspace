'''
Created on 2025/12/01

@author: sin
'''
import matplotlib.pyplot as plt
import networkx as nx

if __name__ == '__main__':

    G = nx.cycle_graph(5)
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    
    for i in range(5):
        G.add_edge(i, (i + 2) % 5)  # Add a new edge for demonstration
    
        ax.clear()  # Clear the axes
        nx.draw(G, ax=ax, with_labels=True)
    
        plt.draw()
        plt.pause(1)  # Pause for a second between updates
    
    plt.ioff()  # Turn off interactive mode
    plt.show()