# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 13:37:01 2023

@author: bommi
"""
#-------------------------------glass file---------------------------------------
                      # step 1  import the file  
import numpy as np
import pandas as pd

df = pd.read_csv('glass.csv')
df

df.shape
df.head()
list(df)

                      #  step 2: EDA 
import matplotlib.pyplot as plt
plt.scatter(df['RI'],df['Na'],color='red')
plt. xlabel('RI')
plt.ylabel('Na')
plt.show()       
                      #  step 3: X and Y variables
Y = df['Type']
X = df.iloc[:,0:9]

                      #  step 4: Data Transformation 
               # Standardization
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss_x = ss.fit_transform(X)
ss_x = pd.DataFrame(ss_x)

                        # Data partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(ss_x,Y,test_size=0.30)

X_train.shape
X_test.shape
                          
          #   knn classifer with its accuracy
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=3)

KNN.fit(X_train,Y_train)

Y_pred_train = KNN.predict(X_train)
Y_pred_test  = KNN.predict(X_test)

           # step6:  metrics
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(Y_train,Y_pred_train)
report = classification_report(Y_test,Y_pred_test)

print("Accuracy:", accuracy.round(2))
print("classification report:\n", report)

#==============================================================================
#---------------------------Zoo file------------------------------------------------
                      # step 1  import the file  
import numpy as np
import pandas as pd

df = pd.read_csv('Zoo.csv')
df

df.shape
df.head()
list(df)
df.dtypes
                      #  step 2: EDA 
import matplotlib.pyplot as plt
plt.scatter(df['milk'],df['hair'],color='red')
plt. xlabel('milk')
plt.ylabel('hair')
plt.show()       


from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
df['animal name']=LE.fit_transform(df['animal name'])

                      #  step 3: X and Y variables
Y = df['animal name']
X = df.iloc[:, 1:]
                      #  step 4: Data Transformation 
                   # Standardization
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss_x = ss.fit_transform(X)
ss_x = pd.DataFrame(ss_x)

                        # Data partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(ss_x,Y,test_size=0.30)
           # Convert DataFrames to NumPy arrays
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
                          
              #   knn classifer with its accuracy
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=5)

KNN.fit(X_train,Y_train)

Y_pred_train = KNN.predict(X_train)
Y_pred_test  = KNN.predict(X_test)

               # step6:  metrics
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(Y_train,Y_pred_train)
report = classification_report(Y_test,Y_pred_test)

print("Accuracy:", accuracy.round(2))
print("classification report:\n", report)
