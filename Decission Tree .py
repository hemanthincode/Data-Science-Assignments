# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 15:21:07 2023

@author: bommi
"""
#-------------------------------------Company_Data----------------------------------------
                       # Load the dataset
import pandas as pd
df = pd.read_csv('Company_Data.csv')
df
print(df.head())

                  # Convert 'Sale' to a categorical variable
df['Sale'] = pd.cut(df['Sales'], bins=[-float('inf'), 8, float('inf')], labels=['Low', 'High'])

                    # Features and target variable
X = df.drop(['Sales', 'Sale'], axis=1)
y = df['Sale']
                              #  label encoding 
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

for column in X.select_dtypes(include=['object']).columns:
    X[column] = label_encoder.fit_transform(X[column])
               # Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                 # Build and train the Decision Tree model
from sklearn.tree import DecisionTreeClassifier, export_text
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

y_pred = dt_model.predict(X_test)

                       # metrics
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
tree_rules = export_text(dt_model, feature_names=list(X.columns))

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)
print("Decision Tree Rules:\n", tree_rules)

##########################################################################################
#---------------------------------Fraud_check---------------------------------------------

                    # Load the dataset
import pandas as pd
df = pd.read_csv('Fraud_check.csv')
df
print(df.head())

df['Taxable.Income'] = df['Taxable.Income'].apply(lambda x: 'Risky' if x <= 30000 else 'Good')

X = df.drop('Taxable.Income', axis=1)
y = df['Taxable.Income']

X = pd.get_dummies(X)
               # Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                 # Build and train the Decision Tree model
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

y_pred = dt_model.predict(X_test)
                            #metrics
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)
