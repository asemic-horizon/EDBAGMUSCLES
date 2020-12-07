import aiosql, sqlite3
import numpy as np
import kmapper as km
from sklearn.preprocessing import *
from sklearn.decomposition import *
from hdbscan import HDBSCAN
from sklearn.cluster import *
from train_feats import get_data
num = 250_000; data = get_data(num)
lens =[0,1]
n_cubes=64
scaler=PowerTransformer()
clusterer=HDBSCAN(metric="l2")


mapper = km.KeplerMapper(verbose=1)
projected_data = mapper.project(data, projection=lens, scaler=scaler)
graph = mapper.map(projected_data, data,
    cover=km.Cover(n_cubes=n_cubes), clusterer=clusterer)
#       clusterer=KMeans(n_clusters=2))

# for i in graph['nodes']: 
#     print(i);
#     print([files[j] for j in graph["nodes"][i]])

# Visualize it
desc = f"_{num}_{lens}_{n_cubes}_{clusterer}_{scaler}"
mapper.visualize(graph, path_html=f"km_{desc}_.html",
                 title=f"Jane Street - {num} samples, lens: {lens}, res.: {n_cubes}, {clusterer}, {scaler}")
