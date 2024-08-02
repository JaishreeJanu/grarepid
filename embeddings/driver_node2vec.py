'''
Source: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/node2vec.py
'''

import os
import sys
import json
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from tqdm import tqdm
from subprocess import run

from torch_geometric.datasets import Planetoid, Amazon, CitationFull
from torch_geometric.nn import Node2Vec
import torch_geometric.transforms as T

# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
# dataset_name = sys.argv[1]
# if dataset_name in ["Cora", "CiteSeer", "PubMed"]:
#     dataset = Planetoid(root='./../torch_datasets', name=dataset_name)
# elif dataset_name in ["Computers", "Photo"]:
#     dataset = Amazon(root='./../torch_datasets', name=dataset_name)
# else:
#     dataset = CitationFull(root='./../torch_datasets', name=dataset_name)
# data = dataset[0]


node2vec_meta = []
for graph in tqdm(os.listdir("../random_graphs/rp_sbm")):

    run(["python3", "node2vec.py",
         "--graph", graph,
         ])