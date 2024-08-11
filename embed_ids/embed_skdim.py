import numpy
import skdim
import numpy as np
import os
import sys
import umap
from sklearn.preprocessing import StandardScaler
import json
from tqdm import tqdm

# generate data : np.array (n_points x n_dim). Here a uniformly sampled 5-ball embedded in 10 dimensions
# data = np.load('../embeddings/gat_embeddings/CiteSeer_embeddings.npy')

# estimate global intrinsic dimension

# estimate local intrinsic dimension (dimension in k-nearest-neighborhoods around each point):
# lpca = skdim.id.lPCA().fit_pw(data,
#                               n_neighbors=100,
#                               n_jobs=1)


def check_numpy_float(val):
    if type(val) is np.float64 or type(val) is np.int64:
        print(type(val.item()))
        return val.item()
    else:
        print(type(val))
        return val
# 'random_sage_embeddings', 'random_gcn_embeddings', 'random_gat_embeddings',
embedding_types = ['random_node2vec_embeddings']
for embedding_type in tqdm(embedding_types):
    this_embedding_type_ids = []
    for data_embed in os.listdir(f'../embeddings/{embedding_type}/'):
        print(data_embed)
        data = np.loadtxt(f'../embeddings/{embedding_type}/{data_embed}')
        danco = skdim.id.DANCo().fit(data)
        corrint = skdim.id.CorrInt().fit(data)
        knn = skdim.id.KNN().fit(data)
        lpca = skdim.id.lPCA().fit(data)
        mada = skdim.id.MADA().fit(data)
        mle = skdim.id.MLE().fit(data)
        mom = skdim.id.MOM().fit(data)
        mind_ml = skdim.id.MiND_ML().fit(data)
        tle = skdim.id.TLE().fit(data)

        this_embedding_type_ids.append({'data': data_embed,
                              'danco': check_numpy_float(danco.dimension_),
                              'corrint': check_numpy_float(corrint.dimension_),
                              'knn': check_numpy_float(knn.dimension_),
                                'lpca': check_numpy_float(lpca.dimension_),
                                'tle': check_numpy_float(tle.dimension_),
                                'mada': check_numpy_float(mada.dimension_),
                                'mle': check_numpy_float(mle.dimension_),
                                'mom': check_numpy_float(mom.dimension_),
                                'mind_ml': check_numpy_float(mind_ml.dimension_)})

    with open(f'./all_{embedding_type}_IDs.json', 'w') as fp:
        json.dump(this_embedding_type_ids, fp)