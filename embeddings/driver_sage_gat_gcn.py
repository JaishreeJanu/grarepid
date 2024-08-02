'''
Author: Jaishree Janu
Source: https://mlabonne.github.io/blog/posts/2022-04-06-GraphSAGE.html
'''
import json
import torch
from torch_geometric.loader import NeighborLoader
import os
from sage_gat_gcn import GraphSAGE, GAT, GCN, test, plot_subgraphs, plot_degree
import numpy as np
from tqdm import tqdm

node_classification = []
for graph in tqdm(os.listdir("../random_graphs/rp_sbm/")):
    data = torch.load(f"../random_graphs/rp_sbm/{graph}")
    data.num_classes = len(set(data.y.tolist()))
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

    graphsage_embed, out = graphsage(data.x, data.edge_index)
    print(out)
    print("*******************************")
    graphsage_embed = graphsage_embed.detach().numpy()
    np.save(f'./rpsbm_sage_embeddings/{graph}.npy', graphsage_embed)

    """# Train GCN"""
    # Create GCN
    gcn = GCN(data.x.size(1), 64, data.num_classes)  ## num_features, 64, num_classes
    # Train
    gcn.fit(data, 200)
    # Test
    gcn_acc = f'{test(gcn, data) * 100: .2f}%'
    print(f'\nGCN test accuracy: {test(gcn, data) * 100:.2f}%\n')

    gcn_embed, out = gcn(data.x, data.edge_index)
    print(out)
    print("*******************************")
    gcn_embed = gcn_embed.detach().numpy()
    np.save(f'./rpsbm_gcn_embeddings/{graph}.npy', gcn_embed)

    """# Train GAT"""
    # Create GAT
    gat = GAT(data.x.size(1), 64, data.num_classes)
    # Train
    gat.fit(data, 200)
    # Test
    gat_acc = f'{test(gat, data) * 100: .2f}%'
    print(f'\nGAT test accuracy: {test(gat, data) * 100:.2f}%\n')

    gat_embed, out = gat(data.x, data.edge_index)
    print(out)
    print("*******************************")
    gat_embed = gat_embed.detach().numpy()
    np.save(f'./rpsbm_gat_embeddings/{graph}.npy', gat_embed)

    node_classification.append({
        'graph': graph,
         'sage_acc': sage_acc,
         'gat_acc': gat_acc,
         'gcn_acc': gcn_acc
    })

with open('./rpsbm_classification_accuracies.json', 'w') as fp:
        json.dump(node_classification, fp)

    ##########################################
    # model_path = f'./sage_models/{dataset_name}_sage.pt'
    # torch.save(graphsage, model_path)
    #
    # model_path = f'./gcn_models/{dataset_name}_gcn.pt'
    # torch.save(gcn, model_path)
    #
    # model_path = f'./gat_models/{dataset_name}_gat.pt'
    # torch.save(gat, model_path)

