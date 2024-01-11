# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 10:06:55 2023

@author: bommi
"""
                     # import the file

import pandas as pd
df = pd.read_csv('bank-full.csv')
df

df.dtypes
                     # label encoding
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['y'] = LE.fit_transform(df['y'])
df['job'] = LE.fit_transform(df['job'])
df['marital'] = LE.fit_transform(df['marital'])
df['education'] = LE.fit_transform(df['education'])
df['default'] = LE.fit_transform(df['default'])
df['housing'] = LE.fit_transform(df['housing'])
df['loan'] = LE.fit_transform(df['loan'])
df['contact'] = LE.fit_transform(df['contact'])
df['month'] = LE.fit_transform(df['month'])
df['poutcome'] = LE.fit_transform(df['poutcome'])
df
                        # seperate X and Y
X = df.iloc[:,[0,5,9,11,12,13,14]]
Y = df['y']
                        # StandardScaler
from sklearn.preprocessing import StandardScaler 
SS = StandardScaler()
SS_X = SS.fit_transform(X)
SS_X = pd.DataFrame(SS_X)
SS_X
                         # LogisticRegression
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression()
logr.fit(SS_X,Y)
logr
Y_pred = logr.predict(SS_X)
Y_pred 
                            # metrics
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score
accuracy = accuracy_score(Y,Y_pred) 
conf_matrix= confusion_matrix(Y,Y_pred)    
class_report= recall_score(Y,Y_pred)    
    
print(f'Accuracy: {accuracy:.2f}')
print('\nConfusion Matrix:',conf_matrix)
print('\nClassification Report:\n',class_report)

                       # Visualize Confusion Matrix
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks([0, 1], ['No', 'Yes'])
plt.yticks([0, 1], ['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()