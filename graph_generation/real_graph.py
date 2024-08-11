import sys
import json
from torch_geometric.datasets import Planetoid, CitationFull, Amazon
import torch
from torch_geometric.utils.convert import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from statistics import mean

from graph_visualizations import plot_in_out_degree_distributions, plot_igraph

datasets = ['cora', 'pubmed', 'citeseer', 'cora_citation', 'citeseer_citation',
            'pubmed_citation', 'cora_ml', 'dblp', 'computers']

real_graphs_meta_data = []
for graph in datasets:
    if graph in ['cora', 'pubmed', 'citeseer']:
        dataset = Planetoid(root='../real_graphs/planetoid/', name=graph)
    elif graph in ['cora_citation', 'citeseer_citation', 'pubmed_citation']:
        dataset = CitationFull(root='../real_graphs/citation/', name=graph[:-9])
    elif graph in ['cora_ml', 'dblp']:
        dataset = CitationFull(root='../real_graphs/citation/', name=graph)
    else:
        dataset = Amazon(root='../real_graphs/amazon/', name=graph)

    data = dataset[0]
    # nx_graph = to_networkx(data)
    # degree_cent = nx.degree_centrality(nx_graph)
    # close_cent = nx.closeness_centrality(nx_graph)
    # between_cent = nx.betweenness_centrality(nx_graph)
    # load_cent = nx.load_centrality(nx_graph)
    real_graphs_meta_data.append({
        'graph': graph,
        'num_nodes': data.num_nodes,
        'num_edges': data.num_edges,
        'node_features': dataset.num_features,
        'num_classes': dataset.num_classes,
        'avg_degree': (2* data.num_edges)/(data.num_nodes),
        # 'degree_cent': mean(degree_cent[k] for k in degree_cent),
        # 'close_cent': mean(close_cent[k] for k in close_cent),
        # 'between_cent': mean(between_cent[k] for k in between_cent),
        # 'load_cent': mean(load_cent[k] for k in load_cent),
    })
    # plot_in_out_degree_distributions(data.edge_index, data.x.size(0), graph)
    # plot_igraph(data, dataset_name=graph)

with open('./real_graphs_meta.json', 'w') as fp:
    json.dump(real_graphs_meta_data, fp)