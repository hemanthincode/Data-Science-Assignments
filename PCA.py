# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 15:15:52 2023

@author: bommi
"""


import pandas as pd
df = pd.read_csv("wine.csv")
df
list(df)
df.dtypes
df.shape

X = df.iloc[:,0:]
X.head()

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X =  SS.fit_transform(X)
SS_X

# pca
from sklearn.decomposition import PCA
pca = PCA()

PC = pca.fit_transform(SS_X)

PC_df = pd.DataFrame(PC)
PC_df.head()

PC_X = df.iloc[:,0:3]
PC_X.columns = ['PC1','PC2','PC3'] 
PC_X 

# KMEans Cluster on SS_X
from sklearn.cluster import KMeans
KMeans = KMeans(n_clusters=3,n_init=30)
KMeans.fit(SS_X)
Y = KMeans.fit_predict(SS_X)
df['KMeans_SS_Y'] = pd.DataFrame(Y)
df['KMeans_SS_Y'].value_counts()

KMeans.inertia_

#To identify the best K value from all possible K values
from sklearn.cluster import KMeans
inertia=[]
for i in range(1,11):
    km=KMeans(n_clusters=i,random_state=0)
    km.fit(SS_X)
    inertia.append(km.inertia_)

print(inertia)

import matplotlib.pyplot as plt
plt.plot(range(1,11),inertia,linestyle='--',marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('inertia')
plt.show()

#Hierarchical clustering on SS_X
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
linked = linkage(SS_X, 'ward')
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram on SS_X')
plt.show()

#K.MEANS CLUSTER on PC_X
from sklearn.cluster import KMeans
KMeans = KMeans(n_clusters=3,n_init=30)
KMeans.fit(PC_X)
Y=KMeans.fit_predict(PC_X)
PC_X["Kmeans_PC_Y"]=pd.DataFrame(Y)
PC_X["Kmeans_PC_Y"].value_counts()

KMeans.inertia_

PC_X

#To identify the best K value from all possible K values
from sklearn.cluster import KMeans
inertia=[]
for i in range(1,11):
    km=KMeans(n_clusters=i,random_state=0)
    km.fit(PC_X)
    inertia.append(km.inertia_)
    
print(inertia)
plt.plot(range(1,11),inertia,linestyle='--',marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('inertia')
plt.show()

#Hierarchical clustering on PC_X

from scipy.cluster.hierarchy import dendrogram, linkage
linked = linkage(PC_X, 'ward')
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram on PC_X')
plt.show()

#comparision
comparison_df = pd.DataFrame({'Original': df['KMeans_SS_Y'], 'PCA': PC_X['Kmeans_PC_Y']})
print(comparison_df)


