from subprocess import run

# names = ["PubMed", "Cora", "CiteSeer",
#          "ogbn-arxiv", "ogbn-products",
#          "ogbn-mag"]

# datasets = ["PubMed", "Cora", "CiteSeer"]  ## PLANETOID
# datasets = ["Computers", "Photo"]  ## AMAZON

# datasets = ["Cora", "Cora_ML", "DBLP", "CiteSeer", "PubMed"]  ## CitationFull

# datasets = ["MUTAG", "ENZYMES", "PROTEINS"] # ] ## TUDataset
# datasets = ["Mutagenicity", "AIDS", "Tox21_ARE"] ## TUDataset
datasets = ["SYNTHETIC", "MSRC_9", "MSRC_21", "MSRC_21C", "Fingerprint", "FIRSTMM_DB", "mit_ct1", "mit_ct2",
            "Mutagenicity", "AIDS", "Tox21_ARE", "TWITTER-Real-Graph-Partial"]


num_samples = 100000
pools = 15

for name in datasets:
    run(["python3", "intrinsic_dimension_k_hops.py",
         "--name", name,
         "--pools", str(pools),
         "--num_samples", str(num_samples)])
    
