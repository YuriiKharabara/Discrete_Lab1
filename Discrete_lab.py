import random
import networkx as nx
import matplotlib.pyplot as plt

from itertools import combinations, groupby


def gnp_random_connected_graph(num_of_nodes: int,
                               completeness: int,
                               draw: bool = False) -> list[tuple[int, int]]:
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi 
    graph, but enforcing that the resulting graph is conneted
    """

    edges = combinations(range(num_of_nodes), 2)
    G = nx.Graph()
    G.add_nodes_from(range(num_of_nodes))
    
    for _, node_edges in groupby(edges, key = lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < completeness:
                G.add_edge(*e)
                
    for (u,v,w) in G.edges(data=True):
        w['weight'] = random.randint(0,10)
                
    if draw: 
        plt.figure(figsize=(10,6))
        nx.draw(G, node_color='lightblue', 
            with_labels=True, 
            node_size=500)
    
    return G


def get_info():             #Крч я тут заклєпав прст окрему функцію яка бере той граф шо генерується і повертає список сам абчиш чого
    G=gnp_random_connected_graph(10,1,False)
    edges = list(map(lambda x: (x[0], x[1], x[2]['weight']), G.edges.data()))

    nodes = list(G.nodes)
    return (edges, nodes)

def kruskal_algorithm():     #Або йобни цей алгоритм, або якшо хочеш я його йобну сам, а ти почни розбиратись з другим завданням
    pass

def prim_algorithm(graph_info):         #Тут Всьо хуйня шас, я прст розбирався з деталаями алгоритму, не зважай на код
    print(graph_info)с
    used_nodes=set()
    unused_nodes=set(graph_info[1])

    s=graph_info[1][0]
    used_nodes.add(s)

    tree=[[s],[]]

    incident=[]
    for i in graph_info[0]:
        if i[0]==s:
            incident.append(i)

    edge_min=min(incident, key=lambda x: x[2])
    tree[1].append(edge_min)
    used_nodes.add(edge_min[1])
    unused_nodes=unused_nodes-used_nodes


    

def main():
    graph_info=get_info()
    T=prim_algorithm(graph_info)


main()