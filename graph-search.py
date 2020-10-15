# CS386 - Artificial Intelligence Lab
# Lab Assignment 1 - Implementation of Uniform Cost Search and A* search
# Date of Assignemnt : September 9, 2020
# Name : Aditi Saxena
# Roll no. : 180010002
#========================================

#importing required libraries

import networkx as nx
import matplotlib.pyplot as plt
from queue import PriorityQueue
import time
#=======================================

# function to print the path and the shortest distance after the search is complete

def result(parent,S,E):
    print("The shortest path is ")

    #storing the path in a list
    path = []
    path.append(E)
    out = parent[E][0]
    while(out != S):
        path.append(out)
        out = parent[out][0]
    path.append(S)

    #traversing the list to print the required path
    for i in range(1,len(path)):
        print(path[len(path)-i]," --> ",end=" ")
    print(E)

    #printing the corresponding shortest distance travelled
    print("The distance travelled is ",parent[E][1])
#=========================================

# function to implement the Uniform Cost Search

def UCS(S,E,G):
    #defining required data structures
    visited = []
    parent = {}
    q = PriorityQueue()

    #adding the start node
    visited.append(S)
    q.put((0,S))
    out =q.get()
    explored = out[1]

    #iteration to explore and expand the node with the lowest cumulative path distance.
    while(explored != E):
        for i in G.neighbors(explored):
            if i not in visited:
                q.put(((out[0]+G[explored][i]['weight']),i))
                if i not in parent.keys() or parent[i][1]>G[explored][i]['weight']+out[0]:
                    parent[i] = [explored,G[explored][i]['weight']+out[0]]
        out = q.get()
        explored = out[1]
        visited.append(explored)

    #printing the result
    result(parent,S,E)
#===========================================

# function to implement the A* search

def Astar(S,E,G):
    #storing the given heuristic values in a dictionary
    h = {'Pernem': 4 , 'Mapusa': 7 , 'Bicholim': 7.25 , 'Panaji':8.5 , 'Ponda': 5 , 'Dharbandora' : 4.75 , 'Valpoi': 2.85 , 'Sanguem': 9.75 , 'Quepem': 3.65 , 'Margao': 5.95 , 'Marmugao': 2.2 , 'Chaudi': 1 , 'Canacona': 2.3 , 'Vasco-de-gama': 1.25}
   
    #end node has a heuristic value zero
    h[E] = 0

    #defining required data structures
    visited = []
    parent = {}
    q = PriorityQueue()

    #adding the start node
    visited.append(S)
    q.put((0+h[S],S))
    out =q.get()
    explored = out[1]

    #iteration to explore and expand the node with the lowest (cumulative path + heuristic value)
    while(explored != E):
        for i in G.neighbors(explored):
            if i not in visited:
                q.put(((out[0]+G[explored][i]['weight'])-h[explored]+h[i],i))
                if i not in parent.keys() or parent[i][1]>G[explored][i]['weight']+out[0]-h[explored]:
                    parent[i] = [explored,G[explored][i]['weight']+out[0]-h[explored]]
        out = q.get()
        explored = out[1]
        visited.append(explored)

    #printing the result
    result(parent,S,E)

#===========================================

# function to visualize the graph nodes and edges

def visualize(G):
    print("Visualizing the graph...Close the image to continue.")
    time.sleep(2)

    #Setting initial parameters
    labels = nx.get_edge_attributes(G,'weight')
    pos = nx.get_node_attributes(G,'pos')

    #display the image
    nx.draw(G,pos,with_labels=True)
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
    plt.savefig("graph.png")
    plt.show()
#============================================

# function to build the graph from the given data

def build():
    #building the graph
    goa = nx.Graph()

    #adding the nodes
    talukas = ["Pernem","Mapusa","Bicholim","Panaji","Ponda","Valpoi","Dharbandora","Sanguem","Margao","Marmugao","Vasco-de-gama","Quepem","Chaudi","Canacona"]
    goa.add_nodes_from(talukas)
    
    #adding edge weights
    weights = [("Pernem","Mapusa",17.2),("Pernem","Bicholim",28),("Mapusa","Bicholim",20.3),("Mapusa","Panaji",16),("Panaji","Ponda",31.4),("Ponda","Bicholim",32.6),("Bicholim","Dharbandora",33.7),("Bicholim","Valpoi",21.7),("Valpoi","Dharbandora",27.8),("Dharbandora","Sanguem",26.4),("Ponda","Margao",17.6),("Margao","Marmugao",30.6),("Marmugao","Vasco-de-gama",2),("Ponda","Quepem",30),("Ponda","Sanguem",32.4),("Quepem","Chaudi",33.7),("Chaudi","Canacona",2.8),("Quepem","Sanguem",13.2),("Sanguem","Canacona",43.2)]
    goa.add_weighted_edges_from(weights)
    
    #adding positions for visualization
    goa.nodes["Pernem"]['pos'] = (0,5)
    goa.nodes["Mapusa"]['pos'] = (2,3)
    goa.nodes["Bicholim"]['pos'] = (2,7)
    goa.nodes["Panaji"]['pos'] = (4,3)
    goa.nodes["Ponda"]['pos'] = (5,5)
    goa.nodes["Valpoi"]['pos'] = (4,9)
    goa.nodes["Dharbandora"]['pos'] = (4,7)
    goa.nodes["Sanguem"]['pos'] = (6,7)
    goa.nodes["Margao"]['pos'] = (6,3)
    goa.nodes["Marmugao"]['pos'] = (4,1)
    goa.nodes["Vasco-de-gama"]['pos'] = (2,1)
    goa.nodes["Quepem"]['pos'] = (8,5)
    goa.nodes["Chaudi"]['pos'] = (9,4)
    goa.nodes["Canacona"]['pos'] = (10,5)
    return goa
#=============================================

def main():
    #construct the graph
    G = build()
    #visualize the graph
    visualize(G)
    start = input("Enter start location: ")
    end = input("Enter end location: ")
    #Uniform Cost Search
    print("-----Printing result using Uniform Cost Search-----")
    UCS(start,end,G)
    input("Press enter to continue...")
    #A-star search
    print("-----Printing result using A-star Search-----")
    Astar(start,end,G)

if __name__ == "__main__":
    main()
