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

datasets = ['erdos-renyi_n400_p0', 'barabasi-albert_n400_m90', 'newman-watts-strogatz_n1500_p0_k50', 'newman-watts-strogatz_n1500_p00_k90',
            'newman-watts-strogatz_n600_p0_k90', 'barabasi-albert_n400_m50', 'newman-watts-strogatz_n1500_p0_k90', 'powerlaw-cluster_n1500_m90_p006',
            'erdos-renyi_n600_p008', 'powerlaw-cluster_n400_m50_p0', 'powerlaw-cluster_n1500_m50_p0', 'newman-watts-strogatz_n1000_p0_k50',
            'newman-watts-strogatz_n1000_p009_k50', 'powerlaw-cluster_n600_m50_p008', 'barabasi-albert_n600_m90', 'barabasi-albert_n1000_m50',
            'powerlaw-cluster_n600_m90_p0', 'powerlaw-cluster_n400_m90_p001', 'newman-watts-strogatz_n600_p01_k90', 'powerlaw-cluster_n1000_m50_p0',
            'barabasi-albert_n600_m50', 'barabasi-albert_n1000_m90', 'barabasi-albert_n1500_m50', 'powerlaw-cluster_n400_m50_p003',
            'newman-watts-strogatz_n400_p022_k90', 'newman-watts-strogatz_n400_p0_k50', 'erdos-renyi_n1000_p0', 'newman-watts-strogatz_n400_p02_k50',
            'powerlaw-cluster_n1500_m90_p003', 'powerlaw-cluster_n1000_m90_p009', 'newman-watts-strogatz_n600_p0_k50', 'newman-watts-strogatz_n400_p012_k90',
            'erdos-renyi_n1500_p003', 'newman-watts-strogatz_n600_p02_k50', 'newman-watts-strogatz_n1000_p02_k90', 'powerlaw-cluster_n600_m90_p008',
            'powerlaw-cluster_n600_m50_p0', 'powerlaw-cluster_n1500_m50_p006', 'newman-watts-strogatz_n1500_p01_k50', 'erdos-renyi_n1500_p0',
            'powerlaw-cluster_n1000_m50_p03', 'barabasi-albert_n1500_m90', 'erdos-renyi_n600_p0', 'erdos-renyi_n400_p022', 'newman-watts-strogatz_n1000_p0_k90',
            'powerlaw-cluster_n1000_m90_p005', 'powerlaw-cluster_n400_m90_p0', 'erdos-renyi_n1000_p005']

second_slot = ['ca-AstroPh-undirected', 'blog-catalog-undirected', 'ca-CondMat-undirected', 'ca-GrQc-undirected', 'ca-HepPh-undirected', 'cit-HepPh', 'cit-HepTh',
               'facebook-ego-undirected', 'facebook-wall', 'flickr-undirected', 'youtube-undirected']
datasets.extend(second_slot)


# datasets = [
#     "karate_club_graph",
#     "les_miserables_graph",
#     "florentine_families_graph",
#     "cora_ml",
#     "citeseer",
#     "amazon_electronics_photo",
#     "amazon_electronics_computers",
#     "pubmed",
#     "cora",
#     "dblp"
# ]

if __name__ == "__main__":
    from os import sys
    print(datetime.now())
    print("Evaluating natural communities")
    
    print("DATASET,AVG-NCLID,STD-NCLID,CV-NCLID,SKW-NCLID,MIN-NCLID,MAX-NCLID")
    tl = []
    for d in datasets:
        graph = DatasetPool.load(d)
        try:
            graph.remove_selfloop_edges()
            graph = graph.to_undirected()
            print(f"{d}: Get LID")
            nclid = NCLIDEstimator(graph)
            nclid.estimate_lids()
        except:
            continue
        nclids = list(nclid.get_lid_values().values())
        with open(d+'nclids.json', 'w') as fp:
            json.dump(nclid.get_lid_values(), fp)
        nclens = [nclid.nc_len(node[0]) for node in graph.nodes()]
        with open(d+'nclens.txt', 'w') as fp:
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

    print(datetime.now())
