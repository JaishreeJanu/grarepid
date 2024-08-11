import sys
import json
from torch_geometric.datasets import StochasticBlockModelDataset, RandomPartitionGraphDataset
import torch
from torch_geometric.utils.convert import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from statistics import mean
import numpy as np

from graph_visualizations import plot_in_out_degree_distributions, plot_igraph

num_blocks = int(sys.argv[1])  # num of blocks/ clusters
num_channels = int(sys.argv[2])  # num of node features
num_graphs = 15

with open ('./sbm_graphs_meta.json', 'r') as fp:
    sbm_graphs_meta = json.load(fp)
for ith_graph in range(num_graphs):
    block_sizes = torch.LongTensor(num_blocks).random_(6,80)
    print(block_sizes)

    edge_probs = [
        [0.25, 0.001, 0.04, 0.0006, 0.0015, 0.0022, 0.001, 0.013, 0.05, 0.0077, 0.0019, 0.00099],
        [0.001, 0.39, 0.00099, 0.0025, 0.03, 0.0414, 0.0031, 0.001, 0.066, 0.0081, 0.00033, 0.0045],
        [0.04, 0.00099, 0.62, 0.0031, 0.0016, 0.0041, 0.0012, 0.017, 0.06, 0.0025, 0.00089, 0.0055],
        [0.0006, 0.0025, 0.0031, 0.84, 0.011, 0.0015, 0.0011, 0.0018, 0.0091, 0.000189, 0.0044, 0.0084],
        [0.0015, 0.03 , 0.0016, 0.011, 0.99, 0.0021, 0.0022, 0.0051, 0.0048, 0.0052, 0.00038, 0.0029],
        [0.0022, 0.0414, 0.0041, 0.0015, 0.0021, 0.214, 0.0014, 0.005, 0.025, 0.0065, 0.00028, 0.0063],
        [0.001, 0.0031, 0.0012, 0.0011, 0.0022, 0.0014, 0.332, 0.0041, 0.00066, 0.0019, 0.0035, 0.088],
        [0.013, 0.001, 0.017, 0.0018, 0.0051, 0.005, 0.0041, 0.151, 0.0044, 0.0078, 0.034, 0.00094],
        [0.05, 0.066, 0.06, 0.0091, 0.0048, 0.025, 0.00066, 0.0044, 0.178, 0.00543, 0.0067, 0.014],
        [0.0077, 0.0081, 0.0025, 0.000189, 0.0052, 0.0065, 0.0019, 0.0078, 0.00543, 0.65, 0.000321,  0.0093],
        [0.0019, 0.00033, 0.00089, 0.0044, 0.00038, 0.00028, 0.0035, 0.034, 0.0067, 0.000321, 0.345, 0.00511],
        [0.00099, 0.0045, 0.0055, 0.0084, 0.0029, 0.0063, 0.088, 0.00094, 0.014, 0.0093, 0.00511, 0.444]
    ]
    # check_edges = np.array(edge_probs)
    # print(np.array_equal(check_edges, check_edges.T))
    # num_rows = len(edge_probs)
    # num_cols = len(edge_probs[0])
    # print(num_rows == num_cols)
    # for i in range((num_rows)):
    #     for j in range(num_cols):
    #         if edge_probs[i][j] != edge_probs[j][i]:
    #             print(i, j, edge_probs[i][j], edge_probs[j][i])
    #
    # sys.exit(1)   # [row[:num_blocks] for row in edge_probs[:num_blocks]]
    edge_probs = [row[-num_blocks:] for row in edge_probs[-num_blocks:]]  ##  [row[-2:] for row in ll[-2:]]
    sbm_torch = StochasticBlockModelDataset(root='./', num_graphs=1,  block_sizes=block_sizes, edge_probs=edge_probs,
                                            num_channels=num_channels, is_undirected=True)

    sbm_torch = sbm_torch[0]
    sbm_torch.num_classes = num_blocks
    torch.save(sbm_torch, f'../random_graphs/sbm_torch_{ith_graph}_{num_blocks}_{num_channels}')

    num_of_nodes = sbm_torch.x.size(0)
    plot_in_out_degree_distributions(sbm_torch.edge_index, num_of_nodes,
                                     f'sbm_torch_{ith_graph}_{num_blocks}_{num_channels}')
    plot_igraph(sbm_torch, dataset_name=f'sbm_torch_{ith_graph}_{num_blocks}_{num_channels}')

    class_colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'brown', 4: 'orange', 5: 'gray',
                               6: 'purple', 7: 'pink', 8: 'olive', 9: 'cyan', 10: 'maroon', 11: 'magenta'}

    nx_random_sbm = to_networkx(sbm_torch, node_attrs=['x', 'y'], to_undirected=False)
    node_colors = [class_colors[data['y']] for _, data in nx_random_sbm.nodes(data=True)]
    nx.draw(nx_random_sbm, node_color=node_colors, with_labels=True, edge_color='gray')

    degree_cent = nx.degree_centrality(nx_random_sbm)
    close_cent = nx.closeness_centrality(nx_random_sbm)
    between_cent = nx.betweenness_centrality(nx_random_sbm)
    load_cent = nx.load_centrality(nx_random_sbm)
    sbm_graphs_meta.append({
                        'graph': f'sbm_torch_{ith_graph}_{num_blocks}_{num_channels}',
                        'num_nodes': sbm_torch.num_nodes,
                        'num_edges': sbm_torch.num_edges,
                        'node_features': num_channels,
                        'num_classes': sbm_torch.num_classes,
                        # 'diameter': diameter,
                        'degree_cent': mean(degree_cent[k] for k in degree_cent),
                        'close_cent': mean(close_cent[k] for k in close_cent),
                        'between_cent': mean(between_cent[k] for k in between_cent),
                        'load_cent': mean(load_cent[k] for k in load_cent),
                        'all_degree_cent': degree_cent,
                        'all_close_cent':  close_cent,
                        'all_between_cent': between_cent,
                        'all_load_cent': load_cent,
                        # 'eccentric': mean(eccentric[k] for k in eccentric),
                        })

    plt.savefig(f'./sbm_nx_plots/sbm_torch_{ith_graph}_{num_blocks}_{num_channels}.png', format='png')
    plt.clf()

with open('./sbm_graphs_meta.json', 'w') as fp:
    json.dump(sbm_graphs_meta, fp)
#
# graph = torch.load('random_sbm')
# graph = dir(graph)
# for attr in graph:
#     print(attr)

###################################################################################333
## setup combinations of parameters for generating RandomPartitionGraph
param_combs = [[3, 3, 15, 0.00011, 0.4], [3, 3, 15, 0.00055, 0.4], [3, 3, 15, 0.0013, 0.4], [3, 3, 15, 0.0033, 0.4],
              [3, 3, 15, 0.00011, 0.8], [3, 3, 15, 0.00055, 1.4], [3, 3, 15, 0.0013, 2.0], [3, 3, 15, 0.0033, 2.5],
              [3, 3, 15, 0.00033, 0.4], [3, 3, 15, 0.00011, 1.4], [3, 3, 15, 0.00013, 2.4], [3, 3, 15, 0.00033, 3.4],
              [3, 3, 15, 0.00011, 3.4], [3, 3, 15, 0.00011, 2.4], [3, 3, 15, 0.00013, 1.4], [3, 3, 15, 0.00033, 0.4],
              [3, 3, 40, 0.00011, 0.4], [3, 3, 40, 0.00088, 0.4], [3, 3, 40, 0.0023, 0.4], [3, 3, 40, 0.0033, 0.4],
              [3, 3, 40, 0.00022, 0.6], [3, 3, 40, 0.00055, 1.5], [3, 3, 40, 0.0013, 2.5], [3, 3, 40, 0.0033, 3.5],
              [3, 3, 40, 0.00099, 0.4], [3, 3, 40, 0.00044, 0.9], [3, 3, 40, 0.00043, 1.6], [3, 3, 40, 0.00022, 3.4],
              [3, 3, 40, 0.0011, 3.4], [3, 3, 40, 0.00044, 2.5], [3, 3, 40, 0.0093, 1.6], [3, 3, 40, 0.00022, 0.6],
              [3, 3, 40, 0.011, 2.2], [3, 3, 40, 0.044, 1.5], [3, 3, 40, 0.043, 0.6], [3, 3, 40, 0.022, 1.2],
              [5, 5, 20, 0.00011, 0.4], [5, 5, 20, 0.00055, 0.4], [5, 5, 20, 0.0013, 0.4], [5, 5, 20, 0.0033, 0.4],
              [5, 5, 20, 0.00033, 0.4], [5, 5, 20, 0.00066, 1.1], [5, 5, 20, 0.00011, 2.0], [5, 5, 20, 0.00013, 3.0],
              [5, 5, 20, 0.00011, 2.4], [5, 5, 20, 0.00055, 1.4], [5, 5, 20, 0.0013, 0.4], [5, 5, 20, 0.0033, 0.9],
              [5, 5, 20, 0.0011, 0.5], [5, 5, 20, 0.00044, 1.8], [5, 5, 20, 0.0013, 2.4], [5, 5, 20, 0.0033, 3.3],
              [5, 5, 20, 0.011, 1.4], [5, 5, 20, 0.055, 1.6], [5, 5, 20, 0.066, 0.8], [5, 5, 20, 0.033, 0.4],
              [5, 5, 50, 0.00011, 0.5], [5, 5, 50, 0.00055, 0.4], [5, 5, 50, 0.0013, 0.6], [5, 5, 50, 0.0053, 0.3],
              [5, 5, 50, 0.00044, 0.5], [5, 5, 50, 0.00077, 1.4], [5, 5, 50, 0.00013, 2.6], [5, 5, 50, 0.00033, 3.3],
              [5, 5, 50, 0.00011, 3.5], [5, 5, 50, 0.00055, 2.4], [5, 5, 50, 0.00013, 1.6], [5, 5, 50, 0.00033, 0.5],
              [5, 5, 50, 0.0011, 1.5], [5, 5, 50, 0.0055, 0.4], [5, 5, 50, 0.0013, 2.6], [5, 5, 50, 0.0033, 1.3],
              [5, 5, 50, 0.011, 1.8], [5, 5, 50, 0.055, 0.9], [5, 5, 50, 0.013, 1.6], [5, 5, 50, 0.033, 0.3],
              [7, 5, 90, 0.00011, 0.5], [7, 5, 90, 0.00055, 0.4], [7, 5, 90, 0.0013, 0.6], [7, 5, 90, 0.0053, 0.3],
              [5, 5, 50, 0.00033, 0.5], [5, 5, 50, 0.00055, 1.4], [5, 5, 50, 0.0013, 2.6], [5, 5, 50, 0.0053, 3.3],
              [7, 5, 50, 0.0066, 0.5], [7, 5, 50, 0.00077, 1.4], [7, 5, 50, 0.00013, 2.6], [7, 5, 50, 0.00083, 3.3],
              [5, 5, 50, 0.00011, 3.5], [5, 5, 50, 0.00055, 2.4], [5, 5, 50, 0.0013, 1.3], [5, 5, 50, 0.0099, 0.4],
              [5, 5, 50, 0.011, 0.5], [5, 5, 50, 0.055, 1.8], [5, 5, 50, 0.033, 0.8], [5, 5, 50, 0.073, 1.3],
              [5, 5, 130, 0.00011, 0.5], [5, 5, 130, 0.00055, 0.4], [5, 5, 130, 0.0013, 0.6], [5, 5, 130, 0.0053, 0.3],
              [7, 5, 130, 0.00033, 0.4], [7, 5, 130, 0.00055, 1.4], [7, 5, 130, 0.00093, 2.4], [7, 5, 130, 0.00063, 3.4],
              [5, 5, 130, 0.00011, 3.5], [5, 5, 130, 0.00066, 2.4], [5, 5, 130, 0.00093, 1.4], [5, 5, 130, 0.00063, 0.4],
              [5, 5, 130, 0.00011, 1.3], [5, 5, 130, 0.00055, 0.6], [5, 5, 130, 0.0093, 2.4], [5, 5, 130, 0.0063, 3.4],
              [7, 5, 130, 0.031, 0.5], [7, 5, 130, 0.055, 1.9], [7, 5, 130, 0.093, 1.2], [7, 5, 130, 0.063, 2.2],
              [5, 8, 30, 0.00011, 0.5], [5, 8, 30, 0.00055, 0.4], [5, 8, 30, 0.0013, 0.6], [5, 8, 30, 0.0099, 0.3],
              [5, 8, 30, 0.00033, 0.5], [5, 8, 30, 0.00055, 1.4], [5, 8, 30, 0.0013, 2.6], [5, 8, 30, 0.0053, 3.3],
              [5, 8, 30, 0.00011, 3.5], [5, 8, 30, 0.00055, 2.4], [5, 8, 30, 0.0013, 1.6], [5, 8, 30, 0.0053, 0.5],
              [5, 8, 30, 0.00021, 3.5], [5, 8, 30, 0.00099, 2.4], [5, 8, 30, 0.00033, 1.6], [5, 8, 30, 0.00053, 0.5],
              [5, 8, 30, 0.031, 1.5], [5, 8, 30, 0.065, 0.4], [5, 8, 30, 0.013, 1.6], [5, 8, 30, 0.053, 0.5],
              [5, 8, 80, 0.00011, 0.5], [5, 8, 80, 0.00055, 0.4], [5, 8, 80, 0.0013, 0.6], [5, 8, 80, 0.0053, 0.3],
              [7, 8, 80, 0.00033, 0.5], [5, 7, 80, 0.00066, 1.4], [5, 7, 80, 0.00013, 2.6], [5, 7, 80, 0.00053, 3.5],
              [5, 8, 80, 0.00033, 3.5], [5, 8, 80, 0.00066, 2.4], [5, 8, 80, 0.00013, 1.6], [5, 8, 80, 0.00053, 0.5],
              [5, 8, 80, 0.00063, 0.7], [5, 8, 80, 0.0066, 1.4], [5, 8, 80, 0.00044, 2.6], [5, 8, 80, 0.0077, 3.5],
              [5, 8, 80, 0.033, 0.5], [5, 8, 80, 0.066, 1.4], [5, 8, 80, 0.013, 0.8], [5, 8, 80, 0.053, 2.1],
              [5, 8, 140, 0.00011, 0.5], [5, 8, 140, 0.00055, 0.4], [5, 8, 140, 0.0033, 0.6], [5, 8, 140, 0.0053, 0.9],
              [7, 8, 140, 0.00044, 0.5], [7, 8, 140, 0.00055, 1.6], [7, 8, 140, 0.0033, 2.7], [7, 8, 140, 0.0053, 3.7],
              [5, 8, 140, 0.00011, 3.5], [5, 8, 140, 0.00055, 2.4], [5, 8, 140, 0.0033, 1.6], [5, 8, 140, 0.0053, 0.6],
              [5, 8, 140, 0.00011, 2.5], [5, 8, 140, 0.0066, 3.4], [5, 8, 140, 0.0099, 0.6], [5, 8, 140, 0.0077, 0.9],
              [5, 8, 140, 0.011, 2.5], [5, 8, 140, 0.066, 1.4], [5, 8, 140, 0.099, 0.6], [5, 8, 140, 0.077, 1.9],
              [7, 8, 200, 0.00011, 0.5], [7, 8, 200, 0.00055, 0.4], [7, 8, 200, 0.0033, 0.6], [7, 8, 200, 0.0053, 0.9],
              [5, 8, 200, 0.00044, 0.5], [5, 8, 200, 0.00055, 1.4], [5, 8, 200, 0.0033, 2.6], [5, 8, 200, 0.0053, 3.9],
              [5, 8, 200, 0.00011, 3.5], [5, 8, 200, 0.00077, 2.4], [5, 8, 200, 0.0033, 1.6], [5, 8, 200, 0.0099, 0.7],
              [7, 8, 200, 0.00011, 1.5], [7, 8, 200, 0.00044, 0.8], [7, 8, 200, 0.00033, 0.6], [7, 8, 200, 0.00077, 0.9],
              [5, 8, 200, 0.041, 1.5], [5, 8, 200, 0.055, 0.4], [5, 8, 200, 0.033, 2.6], [5, 8, 200, 0.053, 0.9],
              [5, 10, 200, 0.00011, 0.5], [5, 10, 200, 0.00055, 0.4], [5, 10, 200, 0.0033, 0.6], [5, 10, 200, 0.0053, 0.9],
              [9, 10, 200, 0.00044, 0.5], [9, 10, 200, 0.00055, 1.4], [9, 10, 200, 0.0033, 2.6], [9, 10, 200, 0.0053, 3.9],
              [5, 10, 200, 0.00011, 3.5], [5, 10, 200, 0.00066, 2.4], [5, 10, 200, 0.0033, 1.6], [5, 10, 200, 0.0088, 0.7],
              [7, 10, 200, 0.00011, 1.5], [7, 10, 200, 0.00044, 0.9], [7, 10, 200, 0.00033, 0.6], [7, 10, 200, 0.00077, 0.9],
              [7, 10, 200, 0.041, 1.5], [7, 10, 200, 0.055, 0.4], [7, 10, 200, 0.033, 2.6], [7, 10, 200, 0.053, 0.9],
              [9, 10, 100, 0.00011, 0.5], [9, 10, 100, 0.00055, 0.4], [9, 10, 100, 0.0033, 0.6], [9, 10, 100, 0.0053, 0.9],
              [7, 10, 100, 0.00044, 0.5], [7, 10, 100, 0.00055, 1.4], [7, 10, 100, 0.0033, 2.6], [7, 10, 100, 0.0053, 3.9],
              [5, 10, 100, 0.00011, 3.5], [5, 10, 100, 0.00099, 2.4], [5, 10, 100, 0.0033, 1.6], [5, 10, 100, 0.0088, 0.7],
              [7, 10, 100, 0.0011, 1.5], [7, 10, 100, 0.00044, 0.9], [7, 10, 100, 0.0033, 0.6], [7, 10, 100, 0.00077, 0.9],
              [5, 10, 100, 0.041, 1.5], [5, 10, 100, 0.055, 0.4], [5, 10, 100, 0.033, 2.6], [5, 10, 100, 0.053, 0.9],
              [7, 10, 300, 0.00011, 0.5], [7, 10, 300, 0.00055, 0.4], [7, 10, 300, 0.0033, 0.6], [7, 10, 300, 0.0053, 0.9],
              [5, 10, 300, 0.00044, 0.5], [5, 10, 300, 0.00055, 1.4], [5, 10, 300, 0.0044, 2.6], [5, 10, 300, 0.0058, 3.8],
              [7, 10, 300, 0.00022, 3.5], [7, 10, 300, 0.0088, 2.4], [7, 10, 300, 0.0033, 1.6], [7, 10, 300, 0.0053, 0.4],
              [9, 10, 300, 0.022, 1.5], [9, 10, 300, 0.088, 0.9], [9, 10, 300, 0.033, 1.6], [9, 10, 300, 0.053, 3.4],
              ]

print("number of rpsbm graphs:", len(param_combs))
with open('./rpsbm_graphs_meta.json', 'r') as f:
    rp_sbm_dict = json.load(f)  ## list to store graph meta data
for ind, param_comb in enumerate(param_combs):
    num_channels = int(param_comb[0]) ## range: [3-9] # num of node features
    num_classes = int(param_comb[1]) ## range: [3-10]
    num_nodes_per_class = int(param_comb[2])  ## range: [15 - 300]
    node_homophily_ratio = float(param_comb[3])  ## range: [0.0001 - 0.088]
    avg_degree = float(param_comb[4]) ## range: [0.3 - 4.0] #avg number of edges connected to the vertices (2 * E) / V
    random_partition_sbm = RandomPartitionGraphDataset(root='./', num_graphs=1, num_classes=num_classes, num_nodes_per_class=num_nodes_per_class,
                                             node_homophily_ratio=node_homophily_ratio,
                                             average_degree=avg_degree, num_channels=num_channels, is_undirected=True)

    random_partition_sbm = random_partition_sbm[0]
    torch.save(random_partition_sbm, f'../random_graphs/rp_sbm/rpsbm_torch_{ind}')

    num_of_nodes = random_partition_sbm.x.size(0)
    plot_in_out_degree_distributions(random_partition_sbm.edge_index, num_of_nodes,
                                     f'rpsbm_torch_{ind}')
    plot_igraph(random_partition_sbm, dataset_name=f'rpsbm_torch_{ind}')
    class_colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'brown', 4: 'orange', 5: 'gray', 6: 'purple', 7: 'pink', 8: 'olive', 9: 'cyan'}

    nx_random_sbm = to_networkx(random_partition_sbm, node_attrs=['x', 'y'], to_undirected=False)

    ## Find the graph metrics from nx
    degree_cent = nx.degree_centrality(nx_random_sbm)
    close_cent = nx.closeness_centrality(nx_random_sbm)
    between_cent = nx.betweenness_centrality(nx_random_sbm)
    load_cent = nx.load_centrality(nx_random_sbm)
    # eccentric = dict(nx.eccentricity(nx_random_sbm))
    # diameter = nx.diameter(nx_random_sbm)

    node_colors = [class_colors[data['y']] for _, data in nx_random_sbm.nodes(data=True)]
    nx.draw(nx_random_sbm, node_color=node_colors, with_labels=True, edge_color='gray')

    plt.savefig(f'./nx_graph_plots/rpsbm_torch_{ind}.png', format='png')
    plt.clf()

    rp_sbm_dict.append({'graph': f'rpsbm_torch_{ind}',
                        'num_nodes': random_partition_sbm.num_nodes,
                        'num_edges': random_partition_sbm.num_edges,
                        'node_features': num_channels,
                        'num_classes': num_classes,
                        'num_nodes_per_class': num_nodes_per_class,
                        'node_homophily_ratio': node_homophily_ratio,
                        'avg_degree': avg_degree,
                        # 'diameter': diameter,
                        'degree_cent': mean(degree_cent[k] for k in degree_cent),
                        'close_cent': mean(close_cent[k] for k in close_cent),
                        'between_cent': mean(between_cent[k] for k in between_cent),
                        'load_cent': mean(load_cent[k] for k in load_cent),
                        # 'eccentric': mean(eccentric[k] for k in eccentric),
                        })

with open('./rpsbm_graphs_meta.json', 'w') as fp:
    json.dump(rp_sbm_dict, fp)
# print(sbm_torch[0])
# print(random_sbm)
# print(random_sbm.y[random_sbm.y])
# torch.save(random_sbm, './random_sbm')







