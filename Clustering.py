# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 11:39:02 2023

@author: bommi
"""
#------------------------------crime_data--------------------------------------
                                # import the file
import pandas as pd 
df = pd.read_csv('crime_data.csv')
df
list(df)
df.dtypes

import matplotlib.pyplot as plt
plt.scatter(x=df['Assault'],y=df['UrbanPop'])
plt.show()

df.head()
X = df.iloc[:,1:]

# data transformation
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X  = SS.fit_transform(X)

## Forming a group using clusters
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='complete')
Y = cluster.fit_predict(SS_X)
df["Y"] = pd.DataFrame(Y)
df["Y"].value_counts()
df

#cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='average')
#Y = cluster.fit_predict(SS_X)
#df["Y"] = pd.DataFrame(Y)
#df["Y"].value_counts()
#df

#==============================================
# data partition
# fit a model with given X and Y
# KNN
# NB
# DT

#==============================================
# Agglomerative Clustering
#==============================================

# data transformation
X = df.iloc[:,2:5]
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X  = SS.fit_transform(X)
SS_X

## Forming a group using clusters
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='complete')
Y = cluster.fit_predict(SS_X)
df["Y_Agg"] = pd.DataFrame(Y)
df["Y_Agg"].value_counts()
df
#==============================================
# K-Means Clustering
#==============================================

from sklearn.cluster import KMeans
KMeans = KMeans(n_clusters = 5, n_init=30)
KMeans.fit(SS_X)
Y = KMeans.predict(SS_X)
df["Kmeans_Y"] = pd.DataFrame(Y)
df["Kmeans_Y"].value_counts()
df
#KMeans.inertia_
# to identify the best k value from all possible k values
from sklearn.cluster import KMeans
inertia = []

for i in range(1, 11):
    km = KMeans(n_clusters=i,random_state=0)
    km.fit(SS_X)
    inertia.append(km.inertia_)
   
print(inertia)

plt.plot(range(1, 11), inertia)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('inertia')
plt.show()

#==============================================
from sklearn.cluster import DBSCAN
DBSCAN()
dbscan = DBSCAN(eps=0.75, min_samples=3)
dbscan.fit(SS_X)

#Noisy samples are given the label -1.
dbscan.labels_

df["dbscan_labels"] = pd.DataFrame(dbscan.labels_)
df["dbscan_labels"].value_counts()

df_final = df[df["dbscan_labels"] != -1]
df_final.shape

df_final.head()
list(df_final)

df_final

#==========================================================================================
#----------------------------------AIRLINES DATA------------------------------------------
import pandas as pd
df=pd.read_excel("EastWestAirlines.xlsx")
df
df.dtypes
list(df)
df.shape
df.head()

#graph
import matplotlib.pyplot as plt
plt.scatter(df['Balance'],df['Qual_miles'],color='purple')
plt.ylabel("Qual_mile")
plt.xlabel("Balance")
plt.show()


#splitting
X=df.iloc[:,1:]

# standardization
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(X)
SS_X = pd.DataFrame(SS_X)
SS_X

#==============================================
# Agglomerative Clustering
#==============================================
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='complete')
Y = cluster.fit_predict(SS_X)
df["Y"] = pd.DataFrame(Y)
df["Y"].value_counts()
df
#==============================================
# K-Means Clustering
#==============================================
from sklearn.cluster import KMeans
KMeans = KMeans(n_clusters=5,n_init=30)
KMeans.fit(SS_X)
df["Kmeans_Y"]=pd.DataFrame(Y)
df["Kmeans_Y"].value_counts()

KMeans.inertia_

#To identify the best K value from all possible K values
from sklearn.cluster import KMeans
inertia=[]
for i in range(1,11):
    km=KMeans(n_clusters=i,random_state=0)
    km.fit(SS_X)
    inertia.append(km.inertia_)
    
print(inertia)
plt.plot(range(1,11),inertia,linestyle='--',marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('inertia')
plt.show()

#C)DBSCAN=NOT ONLY FORMING THE CLUSTERS BUT ALSO FINDING THE OUTLIERS=========================================

from sklearn.cluster import DBSCAN
DBSCAN()
dbscan = DBSCAN(eps=0.75, min_samples=3)
dbscan.fit(SS_X)

#Noisy samples are given the label -1.
dbscan.labels_

df["dbscan_labels"] = pd.DataFrame(dbscan.labels_)
df["dbscan_labels"].value_counts()

df_final = df[df["dbscan_labels"] != -1]
df_final.shape

df_final.head()
list(df_final)

df_final

