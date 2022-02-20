"""This module conataions functions to create and
show a plot based on the algorythms' performance."""

from matplotlib import pyplot as plt
import json


def extract_performance(path: str) -> list[list, list, list]:
    """Gets data on algorythms' performance from the JSON file.

    Args:
        path (str): Path to the JSON file.

    Returns:
        list[list, list, list]: List of Prim and Kruskal performance,
        and numbers of nodes in the tested graphs.
    """
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    nums_of_nodes = [test['num_of_nodes'] for test in data['Prim']]
    prim_perf = [test['avg_time'] for test in data['Prim']]
    kruskal_perf = [test['avg_time'] for test in data['Kraskal']]

    return prim_perf, kruskal_perf, nums_of_nodes


def plot_kruskal(nodes: list, perf: list):
    """Generates and shows plot for Kruskal algorythm only.

    Args:
        nodes (list): List with counts of nodes.
        perf (list): List with average time for each amount of nodes.
    """
    plt.style.use('ggplot')
    plt.plot(nodes, perf, label='Kruskal', marker='.')
    plt.legend()
    plt.title('Kruskal Performance')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time, s')
    plt.tight_layout()

    plt.show()


def plot_prim(nodes: list, perf: list):
    """Generates and shows plot for Prim algorythm only.

    Args:
        nodes (list): List with counts of nodes.
        perf (list): List with average time for each amount of nodes.
    """
    plt.style.use('ggplot')
    plt.plot(nodes, perf, label='Prim', marker='.')
    plt.legend()
    plt.title('Prim  Performance')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time, s')

    plt.show()


def plot_both(nodes: list, perf: list):
    """Generates and shows plot for Prim and Kruskal algorythms.

    Args:
        nodes (list): List with counts of nodes.
        perf (list): Nested list with average time for each amount of nodes.
    """
    perf_p = perf[0]
    perf_k = perf[1]

    plt.style.use('ggplot')
    plt.plot(nodes, perf_p, label='Prim', marker='.')
    plt.plot(nodes, perf_k, label='Kruskal', marker='.')
    plt.legend()
    plt.title('Prim and Kruskal Performance')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time, s')
    plt.tight_layout()

    plt.show()


def main():
    """Main function, runs the process.
    """
    data = extract_performance('stat.json')

    plot_prim(data[2], data[0])
    plot_kruskal(data[2], data[1])
    plot_both(data[2], data[:2])


if __name__ == "__main__":
    main()
