from subprocess import run

# names = ["PubMed", "Cora", "CiteSeer",
#          "ogbn-arxiv", "ogbn-products",
#          "ogbn-mag"]

datasets = ['ca-AstroPh-undirected', 'blog-catalog-undirected', 'ca-CondMat-undirected', 'ca-GrQc-undirected', 'ca-HepPh-undirected',
                    'ca-HepTh-undirected',
                    'cit-HepPh',
                    'cit-HepTh', 'facebook-ego-undirected', 'facebook-wall', 'flickr-undirected', 'ppi',
                    "cora_ml",
                    "citeseer",
                    "amazon_electronics_photo",
                    "amazon_electronics_computers",
                    "pubmed",
                    "cora",
                    "dblp"]  #'youtube-undirected'


num_samples = 100000
pools = 15

for name in datasets:
    run(["python3", "intrinsic_dimension_k_hops_convert_data.py",
         "--name", name,
         "--pools", str(pools),
         "--num_samples", str(num_samples),
         "--datapath", '../data/'])
    
