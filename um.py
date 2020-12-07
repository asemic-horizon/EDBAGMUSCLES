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
from train_feats import get_data

num_points = 300
num_vals = 100
sample_size = st.number_input(label="sample size", value=300)
which_data = st.radio("Data source",['Jane Street','Uniform 3D'])

if which_data=="Jane Street":
	data = get_data(num=sample_size).T
	data = minmax_scale(data)

	data1 = get_data(num=sample_size//2).T
	data1 = minmax_scale(data1)

else:
	data = np.random.uniform(0,1,(sample_size,3))
	data1 = np.random.uniform(0,1,(sample_size//2,3))


max_radius = st.number_input(label="max radius", value=1.0,step=0.10)
xaxis = np.linspace(0,max_radius,num_vals)

dens0 = np.empty((num_vals,))
dens1 = np.empty((num_vals,))

for ix, radius in enumerate(xaxis):
	graph = nx.Graph(radius_neighbors_graph(data,radius=radius).todense())
	N = graph.number_of_nodes()
	p = graph.number_of_edges()/((N*(N-1))/2)
	dens0[ix] = p

	graph = nx.Graph(radius_neighbors_graph(data1,radius=radius).todense())
	N = graph.number_of_nodes()
	p = graph.number_of_edges()/((N*(N-1))/2)
	dens1[ix] = p


fig, ax = plt.subplots()
ax.plot(xaxis,dens0)
ax.plot(xaxis,dens1)
ax.fill_between(xaxis,dens0,dens1,alpha=0.5)
st.write(fig)

fig, ax = plt.subplots()
ax.plot(xaxis,dens1-dens0,'k')
st.write(fig)