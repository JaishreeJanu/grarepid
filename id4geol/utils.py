import os

import networkx as nx
import numpy as np
import scipy.sparse as sp

import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, to_undirected


def parse_npz(path):
    """
    The function parses npz files and return torch_geometric type Data
    """
    with np.load(path) as loader:
        adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                            loader['adj_shape']).tocoo()
        if 'attr_data' in loader:
            x = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                              loader['attr_shape']) ### In the first run, there was todense() here
            x = torch.from_numpy(x).to(torch.float)
            x[x > 0] = 1
        else:
            x = np.eye(adj.shape[0])
            x = torch.from_numpy(x).to(torch.float)
            x[x > 0] = 1

        edge_index = torch.tensor([adj.row, adj.col], dtype=torch.long)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = to_undirected(edge_index, x.size(0))  # Internal coalesce.

        if 'labels' in loader:
            y = torch.from_numpy(loader['labels']).to(torch.long)
        elif "label" in loader:
            y = None
        else:
            y = None
            print("No Labels in the Dataset")

        return Data(x=x, edge_index=edge_index, y=y)


def parse_npz_v2(path, label="label", to_dense=False, undirectedGraph=False):
    """
    The function parses npz files and return torch_geometric type Data
    """
    with np.load(path) as loader:
        adj_matrix = sp.csr_matrix(
            (loader["adj_data"], loader["adj_indices"], loader["adj_indptr"]),
            shape=loader["adj_shape"],
        )
        nx_graph = nx.from_scipy_sparse_array(
            adj_matrix, create_using=nx.DiGraph if not undirectedGraph else nx.Graph
        )
        # get labels first
        node_attrs = {}
        try:
            labels = loader.get(label, loader[label])
            node_attrs = {i: {"label": label} for i, label in enumerate(labels)}

        except:
            print("WARNING: Labels do not exist for this dataset")

        nx.set_node_attributes(nx_graph, node_attrs)

        return