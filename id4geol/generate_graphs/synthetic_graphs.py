## synthetic_graphs

from torch_geometric.datasets import FakeDataset
import torch
from torch_geometric.utils import barabasi_albert_graph
from torch_geometric.data import Data
from typing import Union, Tuple

num_nodes = [200, 500, 800, 1500]
avg_degree_200 = [50.0, 99.0, 174.0, 259.0, 690.0]
avg_degree_500 = [150.0, 269.0, 390.0, 501.0, 640.0, 809.0, 1000.0]
avg_degree_800_1500 = [501.0, 803.0, 1100.0, 1403.0, 1590.0, 2000.0]
num_channels = [64, 128, 256]

for avg_deg in avg_degree_200:
    fakegraph_obj_64 = FakeDataset(num_graphs=1, avg_num_nodes=200, avg_degree=avg_deg, num_channels=num_channels[0], edge_dim=0,
                                num_classes=20, is_undirected=True, task='node').generate_data()
    fakegraph_obj_128 = FakeDataset(num_graphs=1, avg_num_nodes=200, avg_degree=avg_deg, num_channels=num_channels[1],
                                edge_dim=0, num_classes=20, is_undirected=True, task='node').generate_data()
    fakegraph_obj_256 = FakeDataset(num_graphs=1, avg_num_nodes=200, avg_degree=avg_deg, num_channels=num_channels[2],
                                edge_dim=0, num_classes=20, is_undirected=True, task='node').generate_data()

    torch.save(fakegraph_obj_64, f'./../../data/synthetic_graphs/fakegraph-{200}-{avg_deg}-{64}')
    torch.save(fakegraph_obj_128, f'./../../data/synthetic_graphs/fakegraph-{200}-{avg_deg}-{128}')
    torch.save(fakegraph_obj_256, f'./../../data/synthetic_graphs/fakegraph-{200}-{avg_deg}-{256}')

for avg_deg in avg_degree_500:
    fakegraph_obj_64 = FakeDataset(num_graphs=1, avg_num_nodes=500, avg_degree=avg_deg, num_channels=num_channels[0], edge_dim=0,
                                num_classes=20, is_undirected=True, task='node').generate_data()
    fakegraph_obj_128 = FakeDataset(num_graphs=1, avg_num_nodes=500, avg_degree=avg_deg, num_channels=num_channels[1],
                                edge_dim=0, num_classes=20, is_undirected=True, task='node').generate_data()
    fakegraph_obj_256 = FakeDataset(num_graphs=1, avg_num_nodes=500, avg_degree=avg_deg, num_channels=num_channels[2],
                                edge_dim=0, num_classes=20, is_undirected=True, task='node').generate_data()

    torch.save(fakegraph_obj_64, f'./../../data/synthetic_graphs/fakegraph-{500}-{avg_deg}-{64}')
    torch.save(fakegraph_obj_128, f'./../../data/synthetic_graphs/fakegraph-{500}-{avg_deg}-{128}')
    torch.save(fakegraph_obj_256, f'./../../data/synthetic_graphs/fakegraph-{500}-{avg_deg}-{256}')

for avg_deg in avg_degree_800_1500:
    fakegraph_obj_64 = FakeDataset(num_graphs=1, avg_num_nodes=800, avg_degree=avg_deg, num_channels=num_channels[0], edge_dim=0,
                                num_classes=20, is_undirected=True, task='node').generate_data()
    fakegraph_obj_128 = FakeDataset(num_graphs=1, avg_num_nodes=800, avg_degree=avg_deg, num_channels=num_channels[1],
                                edge_dim=0, num_classes=20, is_undirected=True, task='node').generate_data()
    fakegraph_obj_256 = FakeDataset(num_graphs=1, avg_num_nodes=800, avg_degree=avg_deg, num_channels=num_channels[2],
                                edge_dim=0, num_classes=20, is_undirected=True, task='node').generate_data()

    torch.save(fakegraph_obj_64, f'./../../data/synthetic_graphs/fakegraph-{800}-{avg_deg}-{64}')
    torch.save(fakegraph_obj_128, f'./../../data/synthetic_graphs/fakegraph-{800}-{avg_deg}-{128}')
    torch.save(fakegraph_obj_256, f'./../../data/synthetic_graphs/fakegraph-{800}-{avg_deg}-{256}')

for avg_deg in avg_degree_800_1500:
    fakegraph_obj_64 = FakeDataset(num_graphs=1, avg_num_nodes=1500, avg_degree=avg_deg, num_channels=num_channels[0], edge_dim=0,
                                num_classes=20, is_undirected=True, task='node').generate_data()
    fakegraph_obj_128 = FakeDataset(num_graphs=1, avg_num_nodes=1500, avg_degree=avg_deg, num_channels=num_channels[1],
                                edge_dim=0, num_classes=20, is_undirected=True, task='node').generate_data()
    fakegraph_obj_256 = FakeDataset(num_graphs=1, avg_num_nodes=1500, avg_degree=avg_deg, num_channels=num_channels[2],
                                edge_dim=0, num_classes=20, is_undirected=True, task='node').generate_data()

    torch.save(fakegraph_obj_64, f'./../../data/synthetic_graphs/fakegraph-{1500}-{avg_deg}-{64}')
    torch.save(fakegraph_obj_128, f'./../../data/synthetic_graphs/fakegraph-{1500}-{avg_deg}-{128}')
    torch.save(fakegraph_obj_256, f'./../../data/synthetic_graphs/fakegraph-{1500}-{avg_deg}-{256}')

# num_nodes = [400, 800]
# num_edges_400 = [79, 127, 269, 375]
# num_edges_800 = [250, 389, 570, 650, 792]
#
# sample_graph = graph_generator.BAGraph(10, 5)
# torch.save(sample_graph, f"./../../data/synthetic_graphs/synth-{10}-{5}.pt")
# data = torch.load(f"./../../data/synthetic_graphs/synth-{10}-{5}.pt")
# #
# for n_edges in num_edges_400:
#     graph_generator.BAGraph(400, n_edges)
#
# for n_edges in num_edges_800:
#     graph_generator.BAGraph(800, n_edges)
#
# probabilities = [0.35, 0.49, 0.70, 0.85, 0.95]
#
# for p in probabilities:
#     graph_generator.ERGraph(400, p)
#
# for p in probabilities:
#     graph_generator.ERGraph(800, p)