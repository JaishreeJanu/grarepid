# -*- coding: utf-8 -*-
"""
Objective: NODE CLASSIFICATION
Original file is located at
    https://colab.research.google.com/drive/1UxtPEbO9ukRfnKdEYxs15i9ixm83lH_h
"""
import sys
import numpy as np
# Visualization
import networkx as nx

import torch
from torch_geometric.datasets import Planetoid, Amazon, CitationFull, TUDataset, GNNBenchmarkDataset
import torch_geometric.transforms as T
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import igraph as ig
from torch_geometric.loader import NeighborLoader, DataLoader

from torch_geometric.utils import degree
from collections import Counter

import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import SAGEConv, GATv2Conv, GCNConv

# dataset_name = sys.argv[1]
# dataset = Planetoid(root='./data', name=dataset_name)
# dataset = Amazon(root='./data', name=dataset_name, transform=T.TargetIndegree())
# dataset = CitationFull(root='./../torch_datasets', name=dataset_name, transform=T.TargetIndegree())
# try:
#     # dataset = GNNBenchmarkDataset(root='../torch_datasets/', name=dataset_name)
#     data = torch.load("../graph_generation/random_sbm")
# except Exception as e:
#     print("Exception in DATA DOWNLOAD: ", e)
#     sys.exit(0)
# # data = dataset[0]



"""# Mini-batching"""

# # Create batches with neighbor sampling
# train_loader = NeighborLoader(
#     data,
#     num_neighbors=[5, 10],
#     batch_size=16,
#     input_nodes=data.train_mask,
# )
#
# # Print each subgraph
# for i, subgraph in enumerate(train_loader):
#     print(f'Subgraph {i}: {subgraph}')

"""# Implement GraphSage vs. GAT vs. GCN"""

class GraphSAGE(torch.nn.Module):
  """GraphSAGE"""
  def __init__(self, dim_in, dim_h, dim_out):
    super().__init__()
    self.sage1 = SAGEConv(dim_in, dim_h)
    self.sage2 = SAGEConv(dim_h, dim_out)
    self.optimizer = torch.optim.Adam(self.parameters(),
                                      lr=0.01,
                                      weight_decay=5e-4)

  def forward(self, x, edge_index):
    h = self.sage1(x, edge_index)
    h = torch.relu(h)
    h = F.dropout(h, p=0.5, training=self.training)
    h = self.sage2(h, edge_index)
    return h, F.log_softmax(h, dim=1)

  def fit(self, train_loader, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = self.optimizer

    self.train()
    for epoch in range(epochs+1):
      total_loss = 0
      acc = 0
      val_loss = 0
      val_acc = 0

      # Train on batches
      for batch in train_loader:
        optimizer.zero_grad()
        _, out = self(batch.x, batch.edge_index)
        loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
        total_loss += loss
        acc += accuracy(out[batch.train_mask].argmax(dim=1),
                        batch.y[batch.train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        val_loss += criterion(out[batch.val_mask], batch.y[batch.val_mask])
        val_acc += accuracy(out[batch.val_mask].argmax(dim=1),
                            batch.y[batch.val_mask])

      # Print metrics every 10 epochs
      if(epoch % 10 == 0):
          print(f'Epoch {epoch:>3} | Train Loss: {loss/len(train_loader):.3f} '
                f'| Train Acc: {acc/len(train_loader)*100:>6.2f}% | Val Loss: '
                f'{val_loss/len(train_loader):.2f} | Val Acc: '
                f'{val_acc/len(train_loader)*100:.2f}%')

class GAT(torch.nn.Module):
  """Graph Attention Network"""
  def __init__(self, dim_in, dim_h, dim_out, heads=8):
    super().__init__()
    self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
    self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=heads)
    self.optimizer = torch.optim.Adam(self.parameters(),
                                      lr=0.005,
                                      weight_decay=5e-4)

  def forward(self, x, edge_index):
    h = F.dropout(x, p=0.6, training=self.training)
    h = self.gat1(x, edge_index)
    h = F.elu(h)
    h = F.dropout(h, p=0.6, training=self.training)
    h = self.gat2(h, edge_index)
    return h, F.log_softmax(h, dim=1)

  def fit(self, data, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = self.optimizer

    self.train()
    for epoch in range(epochs+1):
        # Training
        optimizer.zero_grad()
        _, out = self(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        acc = accuracy(out[data.train_mask].argmax(dim=1),
                       data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
        val_acc = accuracy(out[data.val_mask].argmax(dim=1),
                           data.y[data.val_mask])

        # Print metrics every 10 epochs
        if(epoch % 10 == 0):
            print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc:'
                  f' {acc*100:>6.2f}% | Val Loss: {val_loss:.2f} | '
                  f'Val Acc: {val_acc*100:.2f}%')

class GCN(torch.nn.Module):
  """Graph Convolutional Network"""
  def __init__(self, dim_in, dim_h, dim_out):
    super().__init__()
    self.gcn1 = GCNConv(dim_in, dim_h)
    self.gcn2 = GCNConv(dim_h, dim_out)
    self.optimizer = torch.optim.Adam(self.parameters(),
                                      lr=0.01,
                                      weight_decay=5e-4)

  def forward(self, x, edge_index):
    h = F.dropout(x, p=0.5, training=self.training)
    h = self.gcn1(h, edge_index)
    h = torch.relu(h)
    h = F.dropout(h, p=0.5, training=self.training)
    h = self.gcn2(h, edge_index)
    return h, F.log_softmax(h, dim=1)

  def fit(self, data, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = self.optimizer

    self.train()
    for epoch in range(epochs+1):
        # Training
        optimizer.zero_grad()
        _, out = self(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        acc = accuracy(out[data.train_mask].argmax(dim=1),
                       data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
        val_acc = accuracy(out[data.val_mask].argmax(dim=1),
                           data.y[data.val_mask])

        # Print metrics every 10 epochs
        if(epoch % 10 == 0):
            print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc:'
                  f' {acc*100:>6.2f}% | Val Loss: {val_loss:.2f} | '
                  f'Val Acc: {val_acc*100:.2f}%')

def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()

def test(model, data):
    """Evaluate the model on test set and print the accuracy score."""
    model.eval()
    _, out = model(data.x, data.edge_index)
    acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
    return acc

def plot_degree(data, graph):
  # Get list of degrees for each node
  degrees = degree(data.edge_index[0]).numpy()

  # Count the number of nodes for each degree
  numbers = Counter(degrees)

  # Bar plot
  fig, ax = plt.subplots(figsize=(12, 6))
  ax.set_xlabel('Node degree')
  ax.set_ylabel('Number of nodes')
  plt.bar(numbers.keys(),
          numbers.values(),
          color='#0A047A')
  plt.savefig(f'./sbm_degrees/{graph}')
  plt.clf()

def plot_subgraphs(train_loader, graph):
    fig = plt.figure(figsize=(16,16))
    for idx, (subdata, pos) in enumerate(zip(train_loader, ['221', '222', '223', '224'])):
        plot_degree(subdata, f"{graph}_sub_{idx}")
        G = to_networkx(subdata, to_undirected=True)
        ax = fig.add_subplot(int(pos[0]), int(pos[1]), int(pos[2]))
        ax.set_title(f'Subgraph {idx}')
        plt.axis('off')
        nx.draw_networkx(G,
                        pos=nx.spring_layout(G, seed=0),
                        with_labels=True,
                        node_size=200,
                        node_color=subdata.y,
                        cmap="cool",
                        font_size=10
                        )
    plt.savefig(f'./sbm_subgraphs/{graph}.png')
    plt.clf()

def convert_to_networkx(graph, n_sample=None):

    g = to_networkx(graph, node_attrs=["x"])
    y = graph.y.numpy()

    # if n_sample is not None:
    #     sampled_nodes = random.sample(g.nodes, n_sample)
    #     g = g.subgraph(sampled_nodes)
    #     y = y[sampled_nodes]

    return g, y
def visualize_classification_result(model, graph, dataset_name, type):

    model.eval()
    _, out_logits = model(graph.x, graph.edge_index)
    pred = torch.argmax(out_logits, dim=1)
    corrects = (pred[graph.test_mask] == graph.y[graph.test_mask]).numpy().astype(int)
    test_index = np.arange(len(graph.x))[graph.test_mask.numpy()]
    g, y = convert_to_networkx(graph)
    g_test = g.subgraph(test_index)

    print("yellow node: correct \npurple node: wrong")

    plt.figure(figsize=(9, 7))
    nx.draw_spring(g_test, node_size=30, arrows=False, node_color=corrects)
    plt.savefig(f'./sbm_test_node_classifications_plots/{type}_{dataset_name}.png')

def plot_classification_results_igraph(model, data, dataset_name, type):
    G = ig.Graph(data.x.size(0), data.edge_index.t().tolist())
    model.eval()
    _, out_logits = model(data.x, data.edge_index)
    predicted_labels = torch.argmax(out_logits, dim=1)

    true_labels = data.y.tolist()
    vertex_colors = []
    correct_classi_nodes = []
    incorrect_classi_nodes = []
    # Determine the color for each node based on correctness of classification
    for i in range(data.x.size(0)):
        if true_labels[i] == predicted_labels[i]:
            correct_classi_nodes.append(i)
            vertex_colors.append('green')  # Correct classification
        else:
            incorrect_classi_nodes.append(i)
            vertex_colors.append('red')  # Incorrect classification

    # Define visual style for the graph
    visual_style = {
        "vertex_color": vertex_colors,
        "vertex_label": G.vs.indices,
        "bbox": (600, 600),
        "margin": 20,
        "vertex_size": 30,
    }

    # Plot the graph
    out = ig.plot(G, **visual_style)
    out.save(f"./sbm_node_classfication_ig/{type}_{dataset_name}.png")

    return incorrect_classi_nodes, correct_classi_nodes, predicted_labels, true_labels