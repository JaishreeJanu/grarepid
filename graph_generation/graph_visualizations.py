# Visualization related imports
import matplotlib.pyplot as plt
import networkx as nx
import igraph as ig

import numpy as np
import torch


def plot_in_out_degree_distributions(edge_index, num_of_nodes, dataset_name):
    """
    explicitly calculate only the node degree statistics here,
    """
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()

    assert isinstance(edge_index, np.ndarray), f'Expected NumPy array got {type(edge_index)}.'

    # Store each node's input and output degree (they're the same for undirected graphs such as Cora)
    in_degrees = np.zeros(num_of_nodes, dtype=np.int64)
    out_degrees = np.zeros(num_of_nodes, dtype=np.int64)

    # Edge index shape = (2, E), the first row contains the source nodes, the second one target/sink nodes
    # Note on terminology: source nodes point to target/sink nodes
    num_of_edges = edge_index.shape[1]
    for cnt in range(num_of_edges):
        source_node_id = edge_index[0, cnt]
        target_node_id = edge_index[1, cnt]

        out_degrees[source_node_id] += 1  # source node points towards some other node -> increment its out degree
        in_degrees[target_node_id] += 1  # similarly here

    hist = np.zeros(np.max(out_degrees) + 1)
    for out_degree in out_degrees:
        hist[out_degree] += 1

    fig = plt.figure(figsize=(12, 8), dpi=100)  # otherwise plots are really small in Jupyter Notebook
    fig.subplots_adjust(hspace=0.6)

    plt.subplot(311)
    plt.plot(in_degrees, color='red')
    plt.xlabel('node id')
    plt.ylabel('in-degree count')
    plt.title('Input degree for different node ids')

    plt.subplot(312)
    plt.plot(out_degrees, color='green')
    plt.xlabel('node id')
    plt.ylabel('out-degree count')
    plt.title('Out degree for different node ids')

    plt.subplot(313)
    plt.plot(hist, color='blue')
    plt.xlabel('node degree')
    plt.ylabel('# nodes for a given out-degree')
    plt.title(f'Node out-degree distribution for {dataset_name} dataset')
    plt.xticks(np.arange(0, len(hist), 5.0))

    plt.grid(True)
    plt.savefig(f'./rpsbm_node_degree_plots/{dataset_name}.png')
    plt.clf()

# data = torch.load("./random_partition_sbm")
# num_of_nodes = data.x.size(0)
# plot_in_out_degree_distributions(data.edge_index, num_of_nodes, f'rpsbm_{data}')

def plot_igraph(data, dataset_name):

    if isinstance(data.edge_index, torch.Tensor):
        edge_index_np = data.edge_index.cpu().numpy()

    if isinstance(data.y, torch.Tensor):
        node_labels_np = data.y.cpu().numpy()

    num_of_nodes = len(node_labels_np)
    edge_index_tuples = list(zip(edge_index_np[0, :], edge_index_np[1, :]))  # igraph requires this format

    # Construct the igraph graph
    ig_graph = ig.Graph()
    ig_graph.add_vertices(num_of_nodes)
    ig_graph.add_edges(edge_index_tuples)

    # Prepare the visualization settings dictionary
    visual_style = {}

    # Defines the size of the plot and margins
    # go berserk here try (3000, 3000) it looks amazing in Jupyter!!! (you'll have to adjust the vertex_size though!)
    visual_style["bbox"] = (2000, 2000)
    visual_style["margin"] = 5

    # I've chosen the edge thickness such that it's proportional to the number of shortest paths (geodesics)
    # that go through a certain edge in our graph (edge_betweenness function, a simple ad hoc heuristic)

    # line1: I use log otherwise some edges will be too thick and others not visible at all
    # edge_betweeness returns < 1 for certain edges that's why I use clip as log would be negative for those edges
    # line2: Normalize so that the thickest edge is 1 otherwise edges appear too thick on the chart
    # line3: The idea here is to make the strongest edge stay stronger than others, 6 just worked, don't dwell on it

    edge_weights_raw = np.clip(np.log(np.asarray(ig_graph.edge_betweenness())+1e-16), a_min=0, a_max=None)
    edge_weights_raw_normalized = edge_weights_raw / np.max(edge_weights_raw)
    edge_weights = [w**6 for w in edge_weights_raw_normalized]
    visual_style["edge_width"] = edge_weights

    # A simple heuristic for vertex size. Size ~ (degree / 4) (it gave nice results I tried log and sqrt as well)
    visual_style["vertex_size"] = [deg*3 for deg in ig_graph.degree()]
    cora_label_to_color_map = {0: 'red', 1: 'blue', 2: 'green', 3: 'brown', 4: 'orange', 5: 'gray', 6: 'purple', 7: 'pink', 8: 'olive', 9: 'cyan'}

    visual_style["vertex_color"] = [cora_label_to_color_map[label] for label in node_labels_np]

    # Set the layout - the way the graph is presented on a 2D chart. Graph drawing is a subfield for itself!
    # used "Kamada Kawai" a force-directed method, this family of methods are based on physical system simulation.
    # (layout_drl also gave nice results for Cora)
    visual_style["layout"] = ig_graph.layout_kamada_kawai()

    # print('Plotting results ... (it may take couple of seconds).')
    out = ig.plot(ig_graph, **visual_style)
    out.save(f"./rpsbm_ig_graphs/{dataset_name}.png")