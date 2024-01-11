# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 11:47:05 2023

@author: bommi
"""

#----------------------------------SalaryData_Test----------------------------------------------

                        #  import the file  
import pandas as pd

df = pd.read_csv('SalaryData_Test.csv')
df
  
df.shape
df.head()
list(df)
                      #  X and Y variables
Y = df['Salary']
X = df.iloc[:,0:13:]

       # Assuming 'column_to_drop' is the column you want to drop
X = X.drop('educationno', axis=1)

list(X)

       # Perform label encoding for categorical variables
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for column in X.select_dtypes(include=['object']).columns:
    X[column] = label_encoder.fit_transform(X[column])

                        # Data partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30)

     # Build and train the Naive Bayes model (Gaussian Naive Bayes)
from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model.fit(X_train, Y_train)

            # Make predictions on the test set
Y_pred = nb_model.predict(X_test)
                        
                   #  metrics
from sklearn.metrics import accuracy_score,classification_report
ac_train = accuracy_score(Y_pred,Y_test)
report = classification_report(Y_pred,Y_test)
print("Training Accuracy:", ac_train.round(2))
print("classification_report:\n",report)




#===============================================================================================
#-----------------------------------SalaryData_Train--------------------------------------------

                        #  import the file  
import pandas as pd

df = pd.read_csv('SalaryData_Train.csv')

print(df.head())

                      #  X and Y variables
y = df['Salary']
X = df.drop('Salary', axis=1)

                        # Label Encoding
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for column in X.select_dtypes(include=['object']).columns:
    X[column] = label_encoder.fit_transform(X[column])

                        # Data partition
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

     # Build and train the Naive Bayes model (Gaussian Naive Bayes)
from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)

                   #  metrics
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)
