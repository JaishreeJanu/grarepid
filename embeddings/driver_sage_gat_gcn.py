'''
Author: Jaishree Janu
Source: https://mlabonne.github.io/blog/posts/2022-04-06-GraphSAGE.html
'''
import json
import sys

import torch
from torch_geometric.loader import NeighborLoader
import os
from sage_gat_gcn import *
import numpy as np
from tqdm import tqdm
from collections import Counter

node_classification = []
for graph in tqdm(os.listdir("../random_graphs/")):
    data = torch.load(f"../random_graphs/{graph}")
    print("Class Distribution:", sorted(Counter(data.y.tolist()).items()))

    # data.num_classes = len(set(data.y.tolist()))
    num_nodes = data.x.size(0)
    num_train = round(num_nodes * 0.8)  # 80% training
    num_val = round(num_nodes * 0.1)  # 10% validation
    num_test = num_nodes - num_train - num_val  # 10% testing

    indices = torch.randperm(num_nodes)
    data.train_mask = indices < num_train
    data.val_mask = (indices >= num_train) & (indices < num_train + num_val)
    data.test_mask = indices >= num_train + num_val

    # Create batches with neighbor sampling
    train_loader = NeighborLoader(
        data,
        num_neighbors=[5, 10],
        batch_size=16,
        input_nodes=data.train_mask,
    )

    for i, subgraph in enumerate(train_loader):
        print(f'Subgraph {i}: {subgraph}')
    plot_subgraphs(train_loader, graph)
    plot_degree(data, graph)

    graphsage = GraphSAGE(data.x.size(1), 64, data.num_classes)
    # Train
    graphsage.fit(train_loader, 200)
    # Test
    sage_acc = f'{test(graphsage, data) * 100: .2f}%'
    print(f'\nGraphSAGE test accuracy: {test(graphsage, data) * 100:.2f}%\n')

    visualize_classification_result(graphsage, data, graph, type='sage')
    incorrect_classi_nodes_sage, correct_classi_nodes_sage, predicted_labels_sage, true_labels_sage = (
        plot_classification_results_igraph(graphsage, data, graph, type='sage'))

    graphsage_embed, out = graphsage(data.x, data.edge_index)
    # print(out)
    print("*******************************")
    graphsage_embed = graphsage_embed.detach().numpy()
    np.save(f'./sbm_sage_embeddings/{graph}.npy', graphsage_embed)

    """# Train GCN"""
    # Create GCN
    gcn = GCN(data.x.size(1), 64, data.num_classes)  ## num_features, 64, num_classes
    # Train
    gcn.fit(data, 200)
    # Test
    gcn_acc = f'{test(gcn, data) * 100: .2f}%'
    print(f'\nGCN test accuracy: {test(gcn, data) * 100:.2f}%\n')

    visualize_classification_result(gcn, data, graph, type='gcn')
    incorrect_classi_nodes_gcn, correct_classi_nodes_gcn, predicted_labels_gcn, true_labels_gcn = (
        plot_classification_results_igraph(gcn, data, graph, type='gcn'))

    gcn_embed, out = gcn(data.x, data.edge_index)
    # print(out)
    print("*******************************")
    gcn_embed = gcn_embed.detach().numpy()
    np.save(f'./sbm_gcn_embeddings/{graph}.npy', gcn_embed)

    """# Train GAT"""
    # Create GAT
    gat = GAT(data.x.size(1), 64, data.num_classes)
    # Train
    gat.fit(data, 200)
    # Test
    gat_acc = f'{test(gat, data) * 100: .2f}%'
    print(f'\nGAT test accuracy: {test(gat, data) * 100:.2f}%\n')

    visualize_classification_result(gat, data, graph, type='gat')
    incorrect_classi_nodes_gat, correct_classi_nodes_gat, predicted_labels_gat, true_labels_gat = (
        plot_classification_results_igraph(gat, data, graph, type='gat'))

    gat_embed, out = gat(data.x, data.edge_index)
    # print(out)
    print("*******************************")
    gat_embed = gat_embed.detach().numpy()
    np.save(f'./sbm_gat_embeddings/{graph}.npy', gat_embed)

    node_classification.append({
        'graph': graph,
        'Class_Distribution': sorted(Counter(data.y.tolist()).items()),
         'sage_test_acc': sage_acc,
         'gat_test_acc': gat_acc,
         'gcn_test_acc': gcn_acc,
        'incorrect_classi_nodes_gat': incorrect_classi_nodes_gat,
        'correct_classi_nodes_gat': correct_classi_nodes_gat,
        'predicted_labels_gat': predicted_labels_gat.tolist(),
        'true_labels_gat': true_labels_gat,
        'incorrect_classi_nodes_gcn': incorrect_classi_nodes_gcn,
        'correct_classi_nodes_gcn': correct_classi_nodes_gcn,
        'predicted_labels_gcn': predicted_labels_gcn.tolist(),
        'true_labels_gcn': true_labels_gcn,
        'incorrect_classi_nodes_sage': incorrect_classi_nodes_sage,
        'correct_classi_nodes_sage': correct_classi_nodes_sage,
        'predicted_labels_sage': predicted_labels_sage.tolist(),
        'true_labels_sage': true_labels_sage,
    })


    model_path = f'./sbm_models/{graph}_sage.pt'
    torch.save(graphsage, model_path)

    model_path = f'./sbm_models/{graph}_gcn.pt'
    torch.save(gcn, model_path)

    model_path = f'./sbm_models/{graph}_gat.pt'
    torch.save(gat, model_path)

with open('./sbm_classification_accuracies.json', 'w') as fp:
        json.dump(node_classification, fp)

##########################################


