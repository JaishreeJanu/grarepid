import os.path as osp
import sys

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--graph')

args = parser.parse_args()
graph = args.graph

# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
# dataset = Planetoid(path, name='Cora')
# data = dataset[0]

data = torch.load(f"../random_graphs/rp_sbm/{graph}")
data.num_classes = len(set(data.y.tolist()))

num_nodes = data.x.size(0)
num_train = round(num_nodes * 0.8)  # 80% training
num_val = round(num_nodes * 0.1)    # 10% validation
num_test = num_nodes - num_train - num_val  # 10% testing

indices = torch.randperm(num_nodes)
data.train_mask = indices < num_train
data.val_mask = (indices >= num_train) & (indices < num_train + num_val)
data.test_mask = indices >= num_train + num_val

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Node2Vec(
    data.edge_index,
    embedding_dim=128,
    walk_length=20,
    context_size=10,
    walks_per_node=10,
    num_negative_samples=1,
    p=1.0,
    q=1.0,
    sparse=True,
).to(device)

num_workers = 4 if sys.platform == 'linux' else 0
loader = model.loader(batch_size=128, shuffle=True, num_workers=num_workers)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)


def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def test():
    model.eval()
    z = model()
    acc = model.test(
        train_z=z[data.train_mask],
        train_y=data.y[data.train_mask],
        test_z=z[data.test_mask],
        test_y=data.y[data.test_mask],
        max_iter=150,
    )
    return acc

for epoch in range(1, 101):
    loss = train()
    #acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')


@torch.no_grad()
def plot_points(colors, graph):
    model.eval()
    z = model().cpu().numpy()
    z = TSNE(n_components=2).fit_transform(z)
    y = data.y.cpu().numpy()

    plt.figure(figsize=(8, 8))
    for i in range(data.num_classes):
        plt.scatter(z[: 0], z[: 1], s=20, color=colors[i])
    plt.axis('off')
    # plt.show()
    plt.savefig(f'node2vec_rp_sbm_plots/{graph}.png')


# colors = [
#     '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535', '#ffd700'
# ]
# plot_points(colors, graph)


all_vectors = ""
for tensor in model(torch.arange(data.num_nodes, device=device)):
    s = "\t".join([str(value) for value in tensor.detach().cpu().numpy()])
    all_vectors += s + "\n"
# save the vectors
with open(f"./rpsbm_node2vec_embeddings/{graph}.txt", "w") as f:
    f.write(all_vectors)
# save the labels
with open(f"./rpsbm_node2vec_labels/{graph}.txt", "w") as f:
    f.write("\n".join([str(label) for label in data.y.numpy()]))

model_path = f'./rpsbm_node2vec_models/{graph}.pt'
torch.save(model.state_dict(), model_path)