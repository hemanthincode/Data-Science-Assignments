# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 23:15:59 2023

@author: bommi
"""
                      # import the file
import pandas as pd
df = pd.read_csv("forestfires.csv")
df
                  # Separate features and target variable
X = df.drop("size_category", axis=1)  # Features
y = df["size_category"]  # Target variable

            # Encode categorical variables 'month' and 'day'
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
X['month_encoded'] = label_encoder.fit_transform(X['month'])
X['day_encoded'] = label_encoder.fit_transform(X['day'])

                  # Drop the original categorical columns
X = X.drop(['month', 'day'], axis=1)

          # Map 'small' and 'large' to 0 and 1 for binary classification
y = y.map({'small': 0, 'large': 1})

          # Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

                 # Build and train the SVM model
from sklearn.svm import SVC
svm_model = SVC(kernel='linear', random_state=42)  
svm_model.fit(X_train_scaled, y_train)

                  # Make predictions on the test set
y_pred = svm_model.predict(X_test_scaled)

                       # metrics
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)

##############################################################################################3
#--------------------------------------SalaryData_Test(1)---------------------------------
                             # Load the dataset
import pandas as pd
df = pd.read_csv('SalaryData_Test(1).csv')

print(df.head())
                             # Separate X and y
X = df.drop('Salary', axis=1)
y = df['Salary']
                    # one-hot encoding or label encoding
                    # one-hot encoding
X_encoded = pd.get_dummies(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.svm import SVC
svm_class = SVC(kernel='linear', C=1.0) 

svm_class.fit(X_train_scaled, y_train)

y_pred = svm_class.predict(X_test_scaled)

                              # metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('\nConfusion Matrix:',conf_matrix)
print('\nClassification Report:\n',class_report)

##############################################################################################
#-------------------------------SalaryData_Train(1)----------------------------------------
                     # Load the  dataset
import pandas as pd
df_train = pd.read_csv('SalaryData_Train(1).csv')
df
print(df_train.head())

                            # Separate  X and  y
X_train = df_train.drop('Salary', axis=1)
y_train = df_train['Salary']
                 # one-hot encoding or label encoding
                  # one-hot encoding
X_train_encoded = pd.get_dummies(X_train)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
                    # Standardizing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.svm import SVC
svm_class = SVC(kernel='linear', C=1.0) 

svm_class.fit(X_train_scaled, y_train)

y_pred = svm_class.predict(X_test_scaled)

                              # metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('\nConfusion Matrix:',conf_matrix)
print('\nClassification Report:\n',class_report)
