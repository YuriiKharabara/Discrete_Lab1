import random
import networkx as nx
import matplotlib.pyplot as plt
import time

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
    [(3, 4, 3), (1, 2, 13), (1, 5, 14), (4, 6, 19), (1, 4, 17)]
    """
    # Створюю сортований список ребер за зростянням ваг,
    # сет неізольованих вершин і словник де ключами є кожна вершина,
    # а значеннями лісти вершин, з якими ця вершина зєднана
    E = sorted(graph_info[0], key=lambda x: x[2])
    connected_nodes = set()
    isolated_groups = {n: n for n in graph_info[1]}
    T = list()

    # тут не складно, я хуй зна як описати, кожен рядок це тупо,
    # якщо хош то напиши в тг що цікавить
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
            connected_nodes.update({v1, v2})
            T.append(edge)

    for edge in E:
        if edge not in T:
            v1, v2 = edge[0], edge[1]
            if v2 not in isolated_groups[v1]:
                T.append(edge)
                isolated_groups[v1] += isolated_groups[v2]
                isolated_groups[v2] = isolated_groups[v1]
    return T


def test_algoritms() -> dict:
    """Run both algorithms 1000 times for each humber of nodes,
    return statistics of performance.

    Returns:
        dict: Statistics of performance. Keys are algorithms,
            values are lists of tuples of num_of_nodes and avg time.
    """
    NODES = [5, 10, 15, 20, 30, 50, 75, 100,
             150, 200, 350, 500, 650, 800, 1000]

    stat = {
        'Prim': [],
        'Kraskal': []
    }

    for num_of_nodes in NODES:
        graph_info = get_info(num_of_nodes)

        # Test Prim
        time_taken = 0
        for _ in range(1000):
            start = time.perf_counter()
            prim_algorithm(graph_info)
            end = time.perf_counter()

            time_taken += end - start

        avg_time = time_taken/1000
        stat['Prim'].append((num_of_nodes, avg_time))

        # Test Kraskal
        time_taken = 0
        for _ in range(1):
            start = time.perf_counter()
            kruskal_algorithm(graph_info)
            end = time.perf_counter()

            time_taken += end - start

        avg_time = time_taken/1000
        stat['Kraskal'].append((num_of_nodes, round(avg_time, 10)))

    return stat


def prim_algorithm(graph_info):
    print(graph_info)
    used_nodes = set()
    unused_nodes = set(graph_info[1])

    s = graph_info[1][0]
    used_nodes.add(s)

    tree = [[s], []]

    incident = []
    for i in graph_info[0]:
        if i[0] == s:
            incident.append(i)

    edge_min = min(incident, key=lambda x: x[2])
    tree[1].append(edge_min)
    used_nodes.add(edge_min[1])
    unused_nodes = unused_nodes-used_nodes


def main():
    # graph_info = get_info(10, 1)
    # Tp = prim_algorithm(graph_info)
    # Tk = kruskal_algorithm(graph_info)
    test_algoritms()


if __name__ == "__main__":
    main()
