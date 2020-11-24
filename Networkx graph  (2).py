#!/usr/bin/env python
# coding: utf-8

# In[2]:


import networkx as nx
G = nx.Graph()
G.add_edge('A', 'B', weight=4)
G.add_edge('B', 'D', weight=2)
G.add_edge('A', 'C', weight=3)
G.add_edge('C', 'D', weight=4)
nx.shortest_path(G, 'A', 'D', weight='weight')


# In[3]:


import networkx as nx
# Not necessary, but will be used later:
import numpy as np 
import matplotlib.pyplot as plt 

g = nx.Graph() # create an empty graph
g.add_node("FirstNode")
g.add_edge(1, 2)
g.edges()


# In[4]:


g.nodes()


# In[5]:


nx.draw(g, with_labels=True)


# In[6]:


list_of_edges = [(1,2), (4,6), (2,5)]
g = nx.Graph()
g.add_edges_from(list_of_edges)
print(g.edges())
print(g.nodes())
nx.draw(g, with_labels=True)


# In[7]:


g_c = nx.complement(g)
plt.subplot(121)
nx.draw(g,node_size=150,node_color='#A0CBE2', with_labels=True)
plt.title("The original graph")
plt.subplot(122)
nx.draw(g_c,node_size=150,node_color='#A0CBE2', with_labels=True)
plt.title("The complement graph")


# In[8]:


g_full = nx.compose(g, g_c)
nx.draw(g_full, node_size=150,node_color='#A0CBE2', with_labels=True) # obviously yields a complete graph


# In[9]:


g_famous = nx.florentine_families_graph()
nx.draw(g_famous, node_size=150,node_color='#A0CBE2', with_labels=True) # obviously yields a complete graph


# In[10]:


#Number oof nodes
g_famous.number_of_nodes()
nx.number_of_nodes(g_famous)
#less efficient, in particular for larger graphs
len(g_famous.nodes())


# In[11]:


A = nx.adjacency_matrix(g_famous)
print(A.todense())


# In[12]:


g_famous.number_of_edges()
nx.number_of_edges(g_famous)


# In[13]:


# A graph is connected if every node can be reached from any other node
nx.is_connected(g_famous)


# In[14]:


#The diameter of a graph is the longest distance between two vertices
nx.diameter(g_famous)


# In[15]:


#The average shortest path length is just the average of all the distances in the graph
nx.average_shortest_path_length(g_famous)


# In[16]:


#The density tells is how many of the possible edges in the graph are actually present
nx.density(g_famous)


# In[17]:


#The transitivity measures the clustering of the graph by relating the number of triangles with the number of triples in the grap
nx.transitivity(g_famous)


# In[18]:


#Another measure for the clustering in a graph the the average clustering coefficient
nx.average_clustering(g_famous)


# In[19]:


#The average degree is just the average of all vertex degrees in the graph
np.mean([i[1] for i in g_famous.degree()])


# In[20]:


#Degree distribution
degs = [i[1] for i in g_famous.degree()]
fig, ax = plt.subplots(figsize=(12, 9))
ax.spines["top"].set_visible(False) # Remove plot frame line on the top 
ax.spines["right"].set_visible(False) # Remove plot frame line on the right
ax.get_xaxis().tick_bottom()  # Remove ticks on the bottom
ax.get_yaxis().tick_left()  # Remove the ticks on the left

ax.hist(degs, color="#3F5D7D", bins='auto');


# In[21]:


#Neighborhood
nx.all_neighbors(g_famous, "Strozzi") # use list comprenehsion to show all neighbors


# In[22]:


#Connectedness. Two vertices are connected of there is a path between
nx.has_path(g_famous, "Pazzi", "Barbadori")


# In[23]:


#Distance between two vertices
nx.shortest_path(g_famous, "Pazzi", "Lamberteschi")


# In[24]:


#Clustering coefficient. The clustering coefficient of a single vertex informs us about 
#how well the neighbors of the vertex are themselves connected. 
#The maximum amount of clustering is achieved if all neighbors of the vertex are neighbores
nx.clustering(g_famous, "Strozzi")


# In[25]:


#Degree The degree of  𝑣𝑖  is just the number of adjacent vertices.
#In a weighted network we usually speak of strength, which is the sum of the weights of all edges.
g_famous.degree("Barbadori")


# In[26]:


# More informative is a second order measure, according to which the degrees of a node are weighted by the degree of the connected node. 
#In other words: the connection to a vertex that has many degrees counts more than a 
#connection to a vertex with few connections. This logic leads to the concept of eigenvector centrality 
#because the recursive calculation of importance can be expressed as the problem of finding an eigenvector.

eigen_centralities = nx.eigenvector_centrality(g_famous).values()

fig, ax = plt.subplots(figsize=(12, 9))
ax.spines["top"].set_visible(False) # Remove plot frame line on the top 
ax.spines["right"].set_visible(False) # Remove plot frame line on the right
ax.get_xaxis().tick_bottom()  # Remove ticks on the bottom
ax.get_yaxis().tick_left()  # Remove the ticks on the left
ax.set_title("Eigenvector centrality")

ax.hist(eigen_centralities, color="#3F5D7D", bins='auto');


# In[27]:


#Betweeness Centrality
#Another interpretation of structural importance is the following: 
#a vertex is important when it connects two large communities, 
#which would remain unconnected if the vertex was not there.

between_centralities = nx.betweenness_centrality(g_famous).values()

fig, ax = plt.subplots(figsize=(12, 9))
ax.spines["top"].set_visible(False) # Remove plot frame line on the top 
ax.spines["right"].set_visible(False) # Remove plot frame line on the right
ax.get_xaxis().tick_bottom()  # Remove ticks on the bottom
ax.get_yaxis().tick_left()  # Remove the ticks on the left
ax.set_title("Betweeness centrality")

ax.hist(between_centralities, color="#3F5D7D", bins='auto');


# In[28]:


fig, ax = plt.subplots(figsize=(12, 9))
nx.draw(g_famous, 
        with_labels=True,
        ax=ax, 
        node_size=75, 
        node_color='#b3d1ff', 
        edge_color="#b3e6cc",
        width =2.0,
        stype= "dashed",
        font_size=10.0,
        font_color="#002966",
        alpha=0.75)
ax.set_title("Zachary's Karate Club friendship network")


# In[29]:


#Here are two representations, one where node sizes depend on degree, 
#the other where they depend on eigenvector centrality
fig, axes = plt.subplots(2,1, figsize=(12, 9))
nx.draw(g_famous, 
        with_labels=False,
        pos=nx.spring_layout(g_famous),
        ax=axes[0], 
        node_size=[i[1]*10 for i in g_famous.degree()],
        node_color='#b3d1ff', 
        edge_color="grey",
        alpha=0.95)
axes[0].set_title("Node size dependent on degree")
nx.draw(g_famous, 
        with_labels=False,
        pos=nx.spring_layout(g_famous),
        ax=axes[1], 
        node_size=[i*1500 for i in nx.eigenvector_centrality(g_famous).values()],  
        node_color='#b3d1ff', 
        edge_color="grey",
        alpha=0.95)
axes[1].set_title("Node size dependent on eigenvector centrality");


# In[30]:


#Visualizing graphs is very difficult. 
#The following figure illustrates how different the same graph can look like if different layouts are used.

fig, axes = plt.subplots(2,1, figsize=(12, 9))
nx.draw(g_famous, 
        with_labels=True,
        pos=nx.spring_layout(g_famous),
        ax=axes[0], 
        node_size=75, 
        node_color='#b3d1ff', 
        edge_color="grey",
        width =2.0,
        stype="dashed",
        font_size=10.0,
        font_color="#002966",
        alpha=0.95)
axes[0].set_title("The networks using Spring-Layout")
nx.draw(g_famous, 
        with_labels=True,
        pos=nx.kamada_kawai_layout(g_famous),
        ax=axes[1], 
        node_size=75, 
        node_color='#b3d1ff', 
        edge_color="grey",
        width =2.0,
        stype= "dashed",
        font_size=10.0,
        font_color="#002966",
        alpha=0.95)
axes[1].set_title("The network using the Kamada-Kawai Layout");


# In[31]:


print(nx.clustering(g_famous))


# In[32]:


A = nx.adjacency_matrix(g_famous)


# In[33]:


print(A)


# In[34]:


D = np.diag(np.ravel(np.sum(A,axis=1)))
L = D - A


# In[35]:


print(L)


# In[54]:


import numpy as np
import networkx as nx
import numpy.linalg as la
import scipy.cluster.vq as vq
import matplotlib.pyplot as plt
#Returns two objects, a 1-D array containing the eigenvalues of a, 
#and a 2-D square array or matrix (depending on the input type) of the corresponding eigenvectors (in columns)
l, U = la.eigh(L)
print(l)
#print(U)


# In[55]:


#Getting the Fiedler vector, the eigenvector corresponding to the second smallest eigenvalue of the Laplacian matrix of of the graph
f = U[:,1]


# In[38]:


#Flattened f and get the values sign
labels = np.ravel(np.sign(f))


# In[39]:


labels


# In[50]:


coord = nx.spring_layout(g_famous, iterations=1000)
coord
names = list(g_famous.nodes)
names
coord


# In[71]:


fig = plt.figure()
label = {}
for node in g_famous.nodes():
    #set the node name as the key and the label as its value 
    label[node] = node

nx.draw_networkx_nodes(g_famous,coord,label,node_size=50,node_color=labels)


# In[ ]:





# In[ ]:




