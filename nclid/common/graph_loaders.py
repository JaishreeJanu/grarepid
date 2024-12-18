import os

import networkx as nx
import numpy as np
import scipy.sparse as sp

from common.graph import Graph


def load_from_file(path, label, to_dense=False):
    """
    Loads a graph from a file.

    Parameters
    ----------
    path : string
        Path to the file.
    label: string
        Name of node label.
    to_dense: bool
        Convert node attributes from default sparse matrix to dense matrix
        representation. Needed for some algorithms (pytorch_geometric GAE).

    Returns
    ----------
    - common.graph.Graph: A loaded graph.
    """
    _, ext = os.path.splitext(path)

    if to_dense and ext[1:] == "csv":
        raise NotImplementedError

    f = globals().get("load_" + ext[1:])
    if f == None:
        raise Exception(
            "loader for the extension {} is not implemented".format(ext[1:])
        )
    return f(path, label, to_dense)


def load_csv(path, label):
    """
    Loads a graph from a CSV file.

    Parameters
    ----------
    path : string
        Path to the CSV file.
    label: string
        Name of node label.

    Returns
    ----------
    - common.graph.Graph: A loaded graph.
    """
    raise Exception("Not implemented")


def load_npz(path, label="label", to_dense=False, undirectedGraph=False):
    """
    Loads a graph from a npz file. For included npz files see:
    https://github.com/abojchevski/graph2gauss/blob/master/g2g/utils.py#L479

    Parameters
    ----------
    path : string
        Path to the npz file.
    label: string
        Name of node label.
    to_dense: bool
        Convert node attributes from default sparse matrix to dense matrix
        representation. Needed for some algorithms (pytorch_geometric GAE).

    Returns
    ----------
    - common.graph.Graph: A loaded graph.
    """
    with np.load(path) as loader:
        adj_matrix = sp.csr_matrix(
            (loader["adj_data"], loader["adj_indices"], loader["adj_indptr"]),
            shape=loader["adj_shape"],
        )

        # For now we do not need attributes.
        # attr_matrix = sp.csr_matrix(
        #     (loader["attr_data"], loader["attr_indices"], loader["attr_indptr"]),
        #     shape=loader["attr_shape"],
        # )
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

        # node_attrs = {i: {"label": i} for i in range(1,16)}
        # For now we do not need attributes.
        # if not to_dense:
        #     try:
        #         len(attr_matrix)
        #     except TypeError as e:
        #         if "sparse matrix length" in str(e):
        #             print(
        #                 "WARNING: You have chosen to use sparse representation of this dataset, "
        #                 "which may not work, depending on the embedding. Please consider passing "
        #                 "to_dense=True if you encounter errors regarding sparse matrix length."
        #             )

        # # add attrs
        # for node_id in node_attrs:
        #     if to_dense:
        #         node_attrs[node_id]["attrs"] = attr_matrix[node_id].todense()
        #     else:
        #         node_attrs[node_id]["attrs"] = attr_matrix[node_id]

        nx.set_node_attributes(nx_graph, node_attrs)

        return Graph(nx_graph)
