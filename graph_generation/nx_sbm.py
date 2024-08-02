"""
Desc: the script to generate SBM (Stochastic Block Model) networkx graphs
"""


import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data
import numpy as np
import torch

def numpy_to_graph(A, node_features=None):
    G = nx.from_numpy_array(A)
    if node_features != None:
        for n in G.nodes():
            for k, v in node_features.items():
                G.nodes[n][k] = v[n]
    return G

num_of_nodes = 14
num_of_feats = 14
adj = np.random.randint(0, 2, [num_of_nodes, num_of_nodes])
feats = np.random.uniform(-1, 1, [num_of_nodes, num_of_feats])

G  = numpy_to_graph(adj, node_features={'x': feats})

# data = Data()
# data.edge_index = torch.from_numpy(adj).nonzero().t()
# data.x = torch.from_numpy(feats)

##########################################################################################
# Stochastic Block Model
def generate_sbm(sizes, p_matrix):
    """
    Generate a nx graph using Stochastic Block Model.
    sizes: List of sizes of groups ()
    p_matrix: Probability matrix for edges within & between groups
    """
       # Ensure the probability matrix is the correct size
    assert len(p_matrix) == len(sizes) and  (len(sizes) == len(row) for row in p_matrix)

    # Generate the SBM graph
    G = nx.stochastic_block_model(sizes=sizes, p=p_matrix, nodelist=range(sum(sizes)), seed=42)
    return G


# Define the sizes of the groups and the probability matrix
block_sizes = [5, 10, 15]  # Example sizes of three blocks
probability_matrix = [
    [0.5, 0.1, 0.05],  # Probabilities of edges within block 1 and between blocks 1-2 and 1-3
    [0.1, 0.6, 0.02],  # Probabilities of edges block 2 and blocks 1-3
    [0.05, 0.02, 0.6]  # Probability of edges within 3
]

# Generate the SBM graph
sbm_graph = generate_sbm(block_sizes, probability_matrix)
# Set the node features
features = np.random.uniform(-1, 1, [sum(block_sizes), 10])

print(sbm_graph.nodes(data=True))
x = torch.from_numpy(features)
# Get the edge_index
edge_index = torch.tensor(list(sbm_graph.edges()), dtype=torch.long).t().contiguous()
y = torch.tensor(sum(block_sizes).reshape(1,-1), dtype=torch.float)

# Draw the generated graph
nx.draw(sbm_graph, with_labels=True, node_color='blue', edge_color='gray')
plt.savefig('my_graph.png', format='png')

# data = from_networkx(sbm_graph)
# print(data)


# features = torch.tensor([sbm_graph.nodes[n]['feature'] for n in sbm_graph.nodes], dtype=torch.float)
# labels = torch.tensor([sbm_graph.nodes[n]['label'] for n in sbm_graph], dtype=torch.long)

# y = torch.tensor([data['label'] _, data in sbm_graph.nodes(data=True)], dtype=torch.long)
data = Data(x=x, edge_index=edge_index, y=y)
torch.save(data, "SBM_graph")
print(data)

###########################################################################################

# import torch
# from torch_geometric.data import Data
# from torch_geometric.utils import erdos_renyi_graph, to_networkx, from_networkx
#
# edge_index = erdos_renyi_graph(150, edge_prob=0.007)
# x = torch.randn(150, 32)
# edge_attr = torch.randn(edge_index.size(1), 32)
# data = Data(edge_index=edge_index, x=x, edge_attr=edge_attr)
#
# print(data)
# G = to_networkx(data, node_attrs=['x'], edge_attrs=['edge_attr'])
# data = from_networkx(G)
# print(data)
#
# nx.draw(G, with_labels=True, node_color='blue', edge_color='gray')
# plt.savefig('erdnyi_graph.png', format='png')