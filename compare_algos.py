"""This module contains functions that implement
Kruskal and Prim algorythms and test their performance."""

import random
import networkx as nx
import matplotlib.pyplot as plt
import time
import math
from tqdm import tqdm
import json
import pprint

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

    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < completeness:
                G.add_edge(*e)

    for (u, v, w) in G.edges(data=True):
        w['weight'] = random.randint(0, 10)

    if draw:
        plt.figure(figsize=(10, 6))
        nx.draw(G, node_color='lightblue',
                with_labels=True,
                node_size=500)

    return G


def get_info(num_of_nodes: int, completeness: float = 1) -> tuple:
    """Return a tuple of list graph's edges and list of nodes.

    Args:
        num_of_nodes (int): Number of nodes in the graph.
        completeness (float): Completeness of the graph.

    Returns:
        tuple: List of edges and list of nodes.
    """
    G = gnp_random_connected_graph(num_of_nodes, completeness)
    edges = list(map(lambda x: (x[0], x[1], x[2]['weight']), G.edges.data()))
    nodes = list(G.nodes)

    return edges, nodes


def kruskal_algorithm(graph_info: tuple) -> list:
    """Return minimum spanning tree using kruskal algorithm.

    Args:
        graph_info (tuple): Tuple of list of edges and nodes.

    Returns:
        list: Edges of a minimum spanning tree.

    >>> kruskal_algorithm(([(1, 2, 13), (1, 3, 18), (1, 4, 17), (1, 5, 14), (1\
, 6, 22), (2, 3, 26), (2, 5, 22), (3, 4, 3), (4, 6, 19)], [1, 2, 3, 4, 5, 6]))
    ([(3, 4, 3), (1, 2, 13), (1, 5, 14), (4, 6, 19), (1, 4, 17)], 66)
    """
    E = sorted(graph_info[0], key=lambda x: x[2])
    connected_nodes = set()
    isolated_groups = {}
    T = list()

    for edge in E:
        v1, v2 = edge[0], edge[1]
        if v1 not in connected_nodes or v2 not in connected_nodes:
            if v1 not in connected_nodes and v2 not in connected_nodes:
                isolated_groups[v1] = [v1, v2]
                isolated_groups[v2] = isolated_groups[v1]
            else:
                if v1 in connected_nodes:
                    isolated_groups[v1].append(v2)
                    isolated_groups[v2] = isolated_groups[v1]
                else:
                    isolated_groups[v2].append(v1)
                    isolated_groups[v1] = isolated_groups[v2]
            connected_nodes.add(v1)
            connected_nodes.add(v2)
            T.append(edge)

    for edge in E:
        v1, v2 = edge[0], edge[1]
        if v2 not in isolated_groups[v1]:
            isolated_groups[v1] += isolated_groups[v2]
            gr2 = set(isolated_groups[v2])
            for node in gr2:
                isolated_groups[node] = isolated_groups[v1]

            T.append(edge)

    weight = 0
    for i in T:
        weight += i[2]

    return T, weight


def get_minimal_weigth(graph_edges, connected_nodes, tree):
    """Goes through graph edges and returns one with the smallest weight which suit our conditions

    Args:
        graph_edges (List): List of edges
        connected_nodes (list): Nodes which are already used
        tree (list): List of edges of our tree

    Returns:
        tuple: edge with smallest weight and which suit requirements
    >>> get_minimal_weigth([(1,2,3), (5,4,0), (3,2,2), (3,4, 5), (5,6,0)], [0,1,2], [(1,2,3)])
    (3, 2, 2)
    """
    used_points = set()
    for verticles in connected_nodes:
        edge = min(graph_edges, key=lambda x: x[2] if
                   ((x[0] == verticles or x[1] == verticles) and
                   (x[0] not in connected_nodes or
                    x[1] not in connected_nodes)) else math.inf)
        used_points.add(edge)
    for i in tree:
        if i in used_points:
            used_points.remove(i)
    dont_needed = set()
    for j in used_points:
        if j[0] in connected_nodes and j[1] in connected_nodes:
            dont_needed.add(j)
    used_points = used_points-dont_needed
    edge = min(used_points, key=lambda x: x[2])
    return edge


def prim_algorithm(graph, weight=0):
    """Prim's algorithm

    Args:
        graph (tuple): (list of edges, list of nodes)
        weight (int, optional): weight of the graph, if you already have some weight you can \
                                add it. Defaults to 0.

    Returns:
        tuple: (Tree=list of edges, weight)
    >>> prim_algorithm(([(0, 1, 1), (0, 4, 1), (1, 4, 1), (2, 4, 8), (3, 4, 4)], [0, 1, 2, 3, 4]))
    ([(0, 1, 1), (1, 4, 1), (3, 4, 4), (2, 4, 8)], 14)
    """
    length = len(graph[1])
    connected_nodes = [0]
    tree = []

    while len(connected_nodes) != length:
        edge = get_minimal_weigth(graph[0], connected_nodes, tree)
        if edge == math.inf:
            break
        tree.append(edge)
        if edge[0] not in connected_nodes:
            connected_nodes.append(edge[0])
        if edge[1] not in connected_nodes:
            connected_nodes.append(edge[1])
    for i in tree:
        weight += i[2]
    tree = sorted(tree, key=lambda x: x[2])
    return tree, weight


def test_algoritms(num_of_iterations: int = 100) -> dict:
    """Run both algorithms 1000 times for each humber of nodes,
    return statistics of performance.

    Returns:
        dict: Statistics of performance. Keys are algorithms,
            values are lists of tuples of num_of_nodes and avg time.
    """
    NODES = [5, 10, 20, 50, 100, 200, 500]
    stat = {
        'Prim': [],
        'Kraskal': []
    }

    for num_of_nodes in NODES:
        print('\n{} nodes'.format(num_of_nodes))

        # Test Prim
        time_taken = 0
        for _ in tqdm(range(num_of_iterations)):
            graph_info = get_info(num_of_nodes, completeness=0.25)
            start = time.perf_counter()
            prim_algorithm(graph_info)
            end = time.perf_counter()

            time_taken += end - start

        avg_time = time_taken/num_of_iterations
        stat['Prim'].append(
            {
                'num_of_nodes': num_of_nodes,
                'avg_time': avg_time
            }
        )

        # Test Kruskal
        time_taken = 0
        for _ in tqdm(range(num_of_iterations)):
            graph_info = get_info(num_of_nodes, completeness=0.3)
            start = time.perf_counter()
            kruskal_algorithm(graph_info)
            end = time.perf_counter()

            time_taken += end - start

        avg_time = time_taken/num_of_iterations
        stat['Kraskal'].append(
            {
                'num_of_nodes': num_of_nodes,
                'avg_time': avg_time
            }
        )

    with open('stat.json', 'w', encoding='utf-8') as file:
        json.dump(stat, file, ensure_ascii=False, indent=4)

    pprint.pprint(stat)
    return stat


def compare_weights():
    for _ in tqdm(range(1000)):
        graph = get_info(50, 0.5)
        tp = prim_algorithm(graph)
        tk = kruskal_algorithm(graph)
        if tk[1] != tp[1]:
            print('Graph: ', graph)
            print('Prim:', tp)
            print('Kruskal:', tk)
            print()


if __name__ == "__main__":
    test_algoritms()