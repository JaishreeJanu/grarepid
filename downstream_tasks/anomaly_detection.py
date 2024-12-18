'''
Author: Jaishree Janu
'''
import sys
from collections import Counter
import torch
from pygod.generator import gen_contextual_outlier

from pygod.detector import DOMINANT
from sklearn.metrics import roc_auc_score, average_precision_score
import json
import argparse
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import networkx as nx

parser = argparse.ArgumentParser()
parser.add_argument('--graph')

args = parser.parse_args()
graph = args.graph

def plot_outliers_graph(graph, dataset_name):
    g = to_networkx(graph, node_attrs=["x"])
    y = graph.y.numpy()

    plt.figure(figsize=(9, 7))
    nx.draw_spring(g, node_size=30, arrows=False, node_color=y)
    plt.savefig(f'./outliers_graphs/{dataset_name}.png')

def train_anomaly_detector(model, graph):
    return model.fit(graph)

def eval_anomaly_detector(model, graph):

    outlier_scores = model.decision_function(graph)
    auc = roc_auc_score(graph.y.numpy(), outlier_scores)
    ap = average_precision_score(graph.y.numpy(), outlier_scores)
    print(f'AUC Score: {auc:.3f}')
    print(f'AP Score: {ap:.3f}')

    return auc, ap

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('./sbm_anomaly_results.json', 'r') as fp:
    anomaly_detection_results = json.load(fp)

data = torch.load(f"../random_graphs/{graph}")
num_outliers = int(0.1 * data.num_nodes) ## 10% set as outliers
## Add outliers
data, y_outlier = gen_contextual_outlier(data, n=num_outliers, k=3)
## Keep labels only as True(outlier), False
data.y = y_outlier.bool()
print(data.y)
plot_outliers_graph(data, graph)
model = DOMINANT()

model = train_anomaly_detector(model, data)
auc, ap = eval_anomaly_detector(model, data)
anomaly_detection_results.append({
    'graph': graph,
    'anomaly_roc_auc': auc,
    'anomaly_avg_precision_score': ap
})


with open('./sbm_anomaly_results.json', 'w') as fp:
    json.dump(anomaly_detection_results, fp)