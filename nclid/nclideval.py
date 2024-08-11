"""
GRASPE -- Graphs in Space: Graph Embeddings for Machine Learning on Complex Data

NCLID evaluation

author: svc@dmi.uns.ac.rs
"""
import os

from common.dataset_pool import DatasetPool
from evaluation.lid_eval import NCLIDEstimator

from datetime import datetime

from statistics import mean, stdev
from scipy.stats import skew
import json
from tqdm import tqdm

import torch
from torch_geometric.utils.convert import to_networkx
from common.graph import Graph
from torch_geometric.datasets import Planetoid, OGB_MAG, Amazon, CitationFull, TUDataset

datasets = ["SYNTHETIC", "MSRC_9", "MSRC_21", "MSRC_21C", "Fingerprint", "FIRSTMM_DB", "mit_ct1", "mit_ct2",
            "Mutagenicity", "AIDS", "Tox21_ARE", "MUTAG", "ENZYMES", "PROTEINS",
            "TWITTER-Real-Graph-Partial", "TRIANGLES", "COLORS-3"]

# "MSRC_9", "MSRC_21", "MSRC_21C", "Fingerprint", "FIRSTMM_DB", "mit_ct1", "mit_ct2", "Mutagenicity", "AIDS", "Tox21_ARE", "MUTAG", "ENZYMES", "PROTEINS", "TWITTER-Real-Graph-Partial"
# "mit_ct1", "mit_ct2": Error: "expected sequence of length 4 at dim 1 (got 2)"
if __name__ == "__main__":
    from os import sys
    print(datetime.now())
    print("Evaluating natural communities")

    # if d in {"Cora", "CiteSeer", "PubMed"}:
    #     data = Planetoid(name=d, root="data/")
    # elif d in {"Computers", "Photo"}:
    #     data = Amazon(name=d, root="../torch_datasets/")
    graph_meta_dict = []
    graph_id_dict = []
    graph_nclen_dict = []
    for d in tqdm(os.listdir('../random_graphs/rp_sbm/')):
        data = torch.load(f'../random_graphs/rp_sbm/{d}')

        graph = Graph(to_networkx(data, node_attrs=['x', 'y'], to_undirected=False))

        # graph_meta_dict.append({'graph': d,
        #                         'num_nodes': data.num_nodes,
        #                         'num_edges': data.num_edges,
        #                         'num_features': data.num_node_features,
        #                         'is_undirected': data.is_undirected(),
        #                         'avg_node_degree': 2 * (data.num_edges) / (data.num_nodes)})
        # graph = DatasetPool.load(d)
        graph.remove_selfloop_edges()
        graph = graph.to_undirected()

        try:
            nclid = NCLIDEstimator(graph)
            nclid.estimate_lids()
        except Exception as e:
            print("Exception OCCURRED in NCLID Algo: ", e)
            continue
        nclids = list(nclid.get_lid_values().values())
        graph_id_dict.append({'graph': d,
                              'avg_nclid': mean(nclids),
                              'min_nclid': min(nclids),
                              'max_nclid': max(nclids)})
        nclens = [nclid.nc_len(node[0]) for node in graph.nodes()]

        graph_nclen_dict.append({'graph': d,
                              'avg_nclen': mean(nclens),
                              'min_nclen': min(nclens),
                              'max_nclen': max(nclens)})


    with open(f'./torch_datasets/random_graphs/rp_sbm_nc_LID.json', 'w') as fp:
        json.dump(graph_id_dict, fp)

    with open(f'./torch_datasets/random_graphs/rp_sbm_nclen.json', 'w') as fp:
        json.dump(graph_nclen_dict, fp)
print(datetime.now())
