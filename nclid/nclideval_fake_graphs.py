"""
GRASPE -- Graphs in Space: Graph Embeddings for Machine Learning on Complex Data

NCLID evaluation

author: svc@dmi.uns.ac.rs
"""
import os

from common.dataset_pool import DatasetPool
from evaluation.lid_eval import NCLIDEstimator
from common.graph_loaders import *

from datetime import datetime

from statistics import mean, stdev
from scipy.stats import skew
import json
import logging
import torch
from torch_geometric.utils.convert import to_networkx
from common.graph import Graph

if __name__ == "__main__":
    from os import sys
    logging.info(datetime.now())
    print("Evaluating natural communities")

    print("DATASET,AVG-NCLID,STD-NCLID,CV-NCLID,SKW-NCLID,MIN-NCLID,MAX-NCLID")
    tl = []
    base_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "data", "synthetic_graphs/"
        )
    for d in os.listdir(base_path):
        fake_graph = torch.load(base_path + d)
        graph = Graph(to_networkx(fake_graph))
        try:
            graph.remove_selfloop_edges()
            graph = graph.to_undirected()
            print(f"{d}: Get LID")
            nclid = NCLIDEstimator(graph)
            nclid.estimate_lids()
        except:
            print("**************** comes in except statement *****************")
            continue
        nclids = list(nclid.get_lid_values().values())
        with open('fake_graphs_nclids/'+d+'_nclids.json', 'w') as fp:
            json.dump(nclid.get_lid_values(), fp)
        nclens = [nclid.nc_len(node[0]) for node in graph.nodes()]
        with open('fake_graphs_nclens/'+d+'_nclens.txt', 'w') as fp:
            fp.write(f"{nclens}\n")

        avg_nclid = mean(nclids)
        std_nclid = stdev(nclids)
        cv_nclid  = std_nclid / avg_nclid
        min_nclid = min(nclids)
        max_nclid = max(nclids)
        skw_nclid = skew(nclids)

        s = d + "," + str(avg_nclid) + "," + str(std_nclid) + "," + str(cv_nclid) + "," + str(skw_nclid) + ","
        s += str(min_nclid) + "," + str(max_nclid)

        print(s)

        avg_nclens = mean(nclens)
        std_nclens = stdev(nclens)
        cv_nclens = std_nclens / avg_nclens
        min_nclens = min(nclens)
        max_nclens = max(nclens)
        skw_nclens = skew(nclens)

        t = d + "," + str(avg_nclens) + "," + str(std_nclens) + "," + str(cv_nclens) + "," + str(skw_nclens) + ","
        t += str(min_nclens) + "," + str(max_nclens)
        tl.append(t)

    print("\n")
    print("DATASET,AVG-NCLEN,STD-NCLEN,CV-NCLEN,SKW-NCLEN,MIN-NCLEN,MAX-NCLEN")
    for t in tl:
        print(t)

    logging.info(datetime.now())
