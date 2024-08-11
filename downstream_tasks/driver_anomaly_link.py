from subprocess import run
from tqdm import tqdm
import os


for graph in tqdm(os.listdir("../random_graphs/")):
    run(["python3", "anomaly_detection.py",
         "--graph", graph,
         ])