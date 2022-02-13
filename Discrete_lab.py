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


def get_info():  # Крч я тут заклєпав прст окрему функцію яка бере той граф
    # шо генерується і повертає список сам абчиш чого
    G = gnp_random_connected_graph(10, 1, False)
    edges = list(map(lambda x: (x[0], x[1], x[2]['weight']), G.edges.data()))

    nodes = list(G.nodes)
    return (edges, nodes)


def kruskal_algorithm(graph_info):
    # Створюю сортований список ребер за зростянням ваг,
    # сет неізольованих вершин і словник де ключами є кожна вершина,
    # а значеннями лісти вершин, з якими ця вершина зєднана
    E = sorted(graph_info[0], key=lambda x: x[2])
    connected_nodes = set()
    isolated_groups = dict()
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
        else:
            if v2 not in isolated_groups[v1]:
                tmp = isolated_groups[v1]
                isolated_groups[v1] += isolated_groups[v2]
                isolated_groups[v2] += tmp
                T.append(edge)

    return T


# Тут Всьо хуйня шас, я прст розбирався з деталаями алгоритму, не зважай на код
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
    graph_info = get_info()
    Tp = prim_algorithm(graph_info)
    Tk = kruskal_algorithm(graph_info)


if __name__ == "__main__":
    main()
