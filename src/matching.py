"""
This is an interface to use blossalg, adapted from the __main__.py file in blossalg repo:
    https://github.com/nenb/blossalg/blob/main/src/blossom/__main__.py
"""
# Standard library imports
import csv
import blossom.blossom as blossom


def read_infile(infile):
    node_array = []
    with open(infile) as csvfile:
        for row in csv.reader(csvfile, delimiter=str(",")):
            neighbours = [idx for idx, row in enumerate(row) if row == "1"]
            node_array.append(neighbours)
    if len(node_array) == 0:
        raise SystemExit("Empty graph. Please supply a valid graph.")
    return node_array


def compute_max_matching(node_array):
    # Create node instances, fill node neighbours
    nodelist = [blossom.Node() for _ in range(len(node_array))]
    for idx, node in enumerate(node_array):
        nodelist[idx].neighbors = [nodelist[node] for node in node]

    # Create graph instance, construct graph
    graph = blossom.Graph()
    graph.nodes = {node.name: node for node in nodelist}
    graph.compute_edges()

    # Compute maximum matching
    graph.find_max_matching()
    dict = graph.create_matching_dict()

    return dict


if __name__ == "__main__":

    node_array = read_infile("../data/input.csv")
    matched_dict = compute_max_matching(node_array)
    print(matched_dict)
    matched_dict = compute_max_matching(node_array)
    print(matched_dict)
    matched_dict = compute_max_matching(node_array)
    print(matched_dict)
