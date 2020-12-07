

import numpy as np
import matplotlib.pyplot as plt
import scipy
from persim import plot_diagrams, PersImage
from ripser import ripser, Rips
from train_feats import get_data


sample_size = 5000
num_points = 300
homology_dim = 2
ndims = 5
dims = {i: None for i in range(4)}
dgms = {i: None for i in range(4)}
for i in dgms:
	data = get_data(sample_size)
	dims[i] = np.random.choice(range(data.shape[1]),ndims)
	dgms[i] = Rips(maxdim=homology_dim,n_perm=num_points).fit_transform(data[:,dims[i]])

fig, ax = plt.subplots(nrows = 2, ncols = 2)
for i in dgms:
	plot_diagrams(dgms[i],lifetime=True,ax=ax[i%2][i//2],legend=i==0)
	ax[i%2][i//2].set_xlabel(f"{list(dims[i])}")
fig.savefig("persistence.png")