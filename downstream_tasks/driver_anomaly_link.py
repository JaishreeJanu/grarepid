from subprocess import run
from tqdm import tqdm
import os


for graph in tqdm(os.listdir("../random_graphs/rp_sbm/")):
    run(["python3", "link_prediction.py",
         "--graph", graph,
         ])