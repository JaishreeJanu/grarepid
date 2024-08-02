from multiprocessing import Pool
import os
import pandas as pd
import argparse
from torch_geometric.transforms import SIGN, AddSelfLoops, ToUndirected
from tqdm import tqdm
import numpy as np
# from ogb.nodeproppred import PygNodePropPredDataset as PD
from torch_geometric.datasets import Planetoid, OGB_MAG, Amazon, CitationFull
from torch_geometric.datasets import TUDataset
import json
import warnings
import torch
warnings.filterwarnings('ignore')


# Command Line Arguments
parser = argparse.ArgumentParser()

parser.add_argument("--pools",
                    type=int,
                    default=15)
parser.add_argument("--num_samples",
                    type=int,
                    default=100000)
# parser.add_argument("--name",
#                     type=str,
#                     default="ogbn-arxiv")
parser.add_argument("--k",
                    type=int,
                    default=5)
parser.add_argument("--destination",
                    type=str,
                    default="")

if not os.path.isdir("ids_estimations/random_graphs/rp_sbm"):
    os.makedirs("ids_estimations/random_graphs/rp_sbm")

args = parser.parse_args()
# if not args.destination:
#     destination = "ids_estimations/random_graphs/" + args.name + ".csv"


# Utility functions


def phi_fj(lf,
           j):
    return np.min(lf[j - 1:] - lf[:1 - j])


def Obs_diam(X):
    return np.max(np.max(X, axis=0) - np.min(X, axis=0))


def cost(n, j):
    "Return costs for phi_fj"
    return n - (j - 1)


def gap_costs(n, x, y):
    """
    Return costs for computing phi_f,x+1,...,phi_f,y-1
    if lfs have length n.
    """
    return sum([cost(n, a) for a in range(x + 1, y)])


print("Start with Computation of Agrregated Vectors")
# Start loading Featrues
# if args.name in {"Cora", "CiteSeer", "PubMed"}:
#     data = Planetoid(name=args.name, root="data/")
# if args.name in {"Computers", "Photo"}:
#     data = Amazon(name=args.name, root="data/")
# elif args.name in {"Cora", "Cora_ML", "DBLP", "CiteSeer", "PubMed"}:
#     data = CitationFull(name=args.name, root="data/")
# elif args.name == "ogbn-mag":
#     data = OGB_MAG(preprocess="metapath2vec", root="data")
# else:
#     data = PD(name=args.name, root="data")

# data = TUDataset(root='../torch_datasets/', name=args.name)

graph_meta_dict = []
graph_id_dict = []
# print(f"dataset name {args.name}, no of graphs: {len(data)}")
for graph in tqdm(os.listdir('../random_graphs/rp_sbm/')):
    G = torch.load(f"../random_graphs/rp_sbm/{graph}")
    ## Pick up each graph from the dataset
    # G = data[graph_index]
    if not args.destination:
        destination = f"ids_estimations/random_graphs/rp_sbm/{graph}.csv"
    graph_meta_dict.append({'graph': graph,
                            'num_nodes': G.num_nodes,
                            'num_edges': G.num_edges,
                            'num_features': G.num_node_features,
                            'is_undirected': G.is_undirected(),
                            'avg_node_degree': 2*(G.num_edges/G.num_nodes)})


# if args.name == "ogbn-mag":
#     G = G.to_homogeneous()
    G = AddSelfLoops()(G)
    G = ToUndirected()(G)
    G = SIGN(args.k)(G)

    print("Compute samples")
    n = len(G["x"])
    # else:
    #     n = len(G['num_nodes'])
    #     print(n)
    samples = list(n + 2 - np.geomspace(n, 2, args.num_samples))
    samples = [int(x) for x in samples]
    samples = sorted(list(set(samples)))
    if samples[0] != 2:
        samples = [2] + samples
    if samples[-1] != n:
        samples.append(n)
    S = np.array(samples)
    gaps = S[1:] - S[:-1]

    # Start computations for Dim
    support_values = []
    results = pd.DataFrame()
    phi_js = np.zeros_like(S, dtype=np.float32)
    between_values = []
    X = np.array(G["x"], dtype=np.float32)
    OD = Obs_diam(X)

    for k in range(args.k + 1):
        print("-----------------")
        print("Start with k: ", k)
        if k == 0:
            X = G["x"]
        else:
            X = G["x" + str(k)]

        X = np.array(X, dtype=np.float32)
        X = (1 / OD) * X

        print("Start Creating LFS")


        def create_lf(i):
            return np.sort(X[:, i])


        with Pool(args.pools) as p:
            lfs = [lf for lf in tqdm(p.imap(create_lf, range(X.shape[1])),
                                     total=X.shape[1])]

        print("Compute Support Phis")


        def phi_j(j):
            return np.array([phi_fj(lf, j) for lf in lfs],
                            dtype=np.float32)


        max_phi = 0  # phi_j for the last support phi_j
        max_phis = []  # store al phi_js
        phi_indices = []  # store for each j which features have to be considered

        with Pool(args.pools) as p:
            for i, phi_lfs in enumerate(tqdm(p.imap(phi_j,
                                                    samples),
                                             total=len(samples))):
                phi_indices.append(np.where(phi_lfs > max_phi)[0])
                max_phi = np.max(phi_lfs)
                max_phis.append(max_phi)
                max_phi = np.max([max_phi, phi_js[i]])

        phi_indices = phi_indices[1:]

        print("Compute Bounds and errors")
        support_values.append(np.array(max_phis, dtype=np.float32))
        phi_js = np.max(support_values, axis=0)
        print("Support Sum: ", np.sum(phi_js))
        min_Delta = (np.sum(phi_js[:-1] * gaps) + phi_js[-1]) * (1 / n)
        max_Delta = (np.sum(phi_js[1:] * gaps) + phi_js[0]) * (1 / n)
        max_Dim = 1 / (min_Delta ** 2)
        min_Dim = 1 / (max_Delta ** 2)
        error = (max_Dim - min_Dim) / min_Dim
        print("Error:", error)

        print("Compute Costs")

        pre_costs = np.sum([len(lfs) * cost(n, s) for s in samples])
        full_gap_costs = np.array([gap_costs(n, x, y)
                                   for x, y in zip(S, S[1:])])
        full_gap_costs = np.array([len(s) for s in phi_indices]) * full_gap_costs
        full_gap_costs = np.sum(full_gap_costs)
        costs_full_computation = len(lfs) * ((n ** 2 / 2) - (n / 2))

        skipped_costs = 1 - ((pre_costs + full_gap_costs) / costs_full_computation)
        print("Skipped costs:", skipped_costs)

        print("Make exact computation")


        def phi_j(t):
            i, j, current_lfs, default = t
            gap = range(i + 1, j)
            values = np.array([[phi_fj(lfs[i], j) for i in current_lfs]
                               for j in gap],
                              dtype=np.float32)
            try:
                return np.max(values, axis=1, initial=default)
            except Exception:  # Sometimes values is empty because array was empty
                return []


        max_phis = []
        with Pool(args.pools) as p:
            for d_list in tqdm(p.imap(phi_j,
                                      zip(samples,
                                          samples[1:],
                                          phi_indices,
                                          phi_js)),
                               total=len(phi_indices)):
                max_phis.extend(d_list)
        between_values.append(np.array(max_phis, dtype=np.float32))
        gap_phi_js = np.max(between_values, axis=0)
        all_js = np.sort(np.concatenate([phi_js, gap_phi_js]))
        Delta = np.sum(all_js)
        Delta = (1 / n) * Delta
        Dim = 1 / (Delta ** 2)

        results = results.append({"k": k,
                                  "min_Dim": min_Dim,
                                  "max_Dim": max_Dim,
                                  "error": error,
                                  "Dim": Dim,
                                  "skipped_costs": skipped_costs},
                                 ignore_index=True)
    results.to_csv(destination, mode='a', index=False)

    graph_id_dict.append({'graph': graph,
                          'min_dim': min_Dim,
                          'max_dim': max_Dim,
                          'dim': Dim})
with open('./ids_estimations/rp_sbm_random_meta.json', 'w') as fp:
    json.dump(graph_meta_dict, fp)

with open('./ids_estimations/rp_sbm_random_geol_ID.json', 'w') as fp:
    json.dump(graph_id_dict, fp)