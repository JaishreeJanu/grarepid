from subprocess import run
import os

# names = ["PubMed", "Cora", "CiteSeer",
#          "ogbn-arxiv", "ogbn-products",
#          "ogbn-mag"]

base_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "out_random_graphs_2"
        )
datasets = []
for d in os.listdir(base_path):
    datasets.append(d)
num_samples = 100000
pools = 15

for name in datasets:
    print("------")
    print("------")
    print(name)
    run(["python3", "intrinsic_dimension_k_hops_convert_data_random_sec.py",
         "--name", name,
         "--pools", str(pools),
         "--num_samples", str(num_samples),
         "--datapath", 'out_random_graphs_2/'])
    
