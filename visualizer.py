import networkx as nx
from networkx.drawing.nx_agraph import write_dot


def visualize(graph, output_file):
    g = nx.Graph()

    for key, val in graph.items():
        for to, r in val:
            g.add_edge(key, to, label=r)

    write_dot(g, output_file)
