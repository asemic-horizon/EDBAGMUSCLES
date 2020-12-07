"""
Edge density based estimation of global multiscale clumpiness (EDBAGMUSCLES)
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy
from  sklearn.neighbors import radius_neighbors_graph
from sklearn.preprocessing import minmax_scale
import networkx as nx

num_points = 300
sample_size = st.number_input(label="sample size", value=300)
which_data = st.radio("Data source",['Jane Street','Uniform 3D'])

if which_data=="Jane Street":
	data = get_data(num=sample_size).T
	data = minmax_scale(data)
else:
	data = np.random.uniform(0,1,(sample_size,3))

max_radius = st.number_input(label="max radius", value=1.0,step=0.10)
xaxis = np.linspace(0,max_radius,num_vals)
dens = np.empty((num_vals,))
for ix, radius in enumerate(xaxis):
	graph = nx.Graph(radius_neighbors_graph(data,radius=radius).todense())
	N = graph.number_of_nodes()
	p = graph.number_of_edges()/((N*(N-1))/2)
	dens[ix] = p

fig, ax = plt.subplots()
ax.plot(xaxis,dens)
st.write(fig)