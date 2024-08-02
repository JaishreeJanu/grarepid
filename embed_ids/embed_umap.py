import plotly.express as px
import embed_umap.umap_ as umap
import numpy as np
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


X = np.load('../embeddings/sage_embeddings/Photo_embeddings.npy')

reducer = umap.UMAP()
scaled_penguin_data = StandardScaler().fit_transform(X)

embedding = reducer.fit_transform(scaled_penguin_data)
print(embedding.shape)