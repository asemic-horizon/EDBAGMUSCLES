

import numpy as np
import matplotlib.pyplot as plt
import scipy
from persim import plot_diagrams, PersImage
from ripser import ripser, Rips
from train_feats import get_data

sample_size = 5000
num_points = 300
homology_dim = 2

data = get_data(num=sample_size)
dgm = Rips(maxdim=homology_dim,n_perm=num_points).fit_transform(data)

img_generator = PersImage()

im = img_generator.transform(dgm)

fig, ax = plt.subplots()

img_generator.show(im,ax=ax)

fig.savefig("pim.png")