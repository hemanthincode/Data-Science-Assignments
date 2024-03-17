# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 12:21:03 2023

@author: bommi
"""
import numpy as np
import pandas as pd
df = pd.read_csv("Boston.csv")
df.head()

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

Y = df["medv"]
X = df.drop('medv', axis=1)
df.head(X)
#list(X)
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(X)
SS_X = pd.DataFrame(SS_X)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test  = train_test_split(X,Y)

#=================================
from sklearn.tree import DecisionTreeRegressor
DT = DecisionTreeRegressor(criterion='squared_error')

DT.fit(X_train,Y_train)
Y_pred_train = DT.predict(X_train)
Y_pred_test = DT.predict(X_test)


from sklearn.metrics import mean_squared_error
mse1= np.sqrt(mean_squared_error(Y_train,Y_pred_train))
mse2 = np.sqrt(mean_squared_error(Y_test,Y_pred_test))
print("Training Error", mse1.round(3))
print("Test Error", mse2.round(3))
#====================================================================
# cross validation
#====================================================================

 
training_error = []
test_error = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=i)
    DT.fit(X_train,Y_train)
    Y_pred_train = DT.predict(X_train)
    Y_pred_test  = DT.predict(X_test)
    training_error.append(mean_squared_error(Y_train,Y_pred_train))
    test_error.append(mean_squared_error(Y_test,Y_pred_test))

print("Cross validation training_error:",np.mean(training_error).round(2))
print("Cross validation test_error:",np.mean(test_error).round(2))
print('variance:',np.mean(test_error)-np.mean(training_error).round(2))
#====================================================================
# Bagging Classifier
#====================================================================

from sklearn.ensemble import BaggingRegressor
bag = BaggingRegressor(estimator=DT,n_estimators=100,
                       max_features=0.7,max_samples=0.6)
    
training_accuracy = []
test_accuracy = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=i)
    bag.fit(X_train,Y_train)
    Y_pred_train = bag.predict(X_train)
    Y_pred_test  = bag.predict(X_test)
    training_error.append(np.sqrt(mean_squared_error(Y_train,Y_pred_train)))
    test_error.append(np.sqrt(mean_squared_error(Y_test,Y_pred_test)))


print("Cross validation training_error:",np.mean(training_error).round(2))
print("Cross validation test_error:",np.mean(test_error).round(2))
print('variance:',np.mean(test_error)-np.mean(training_error).round(2))
#====================================================================
# RandomForest Classifier
#====================================================================

from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor(n_estimators=100,max_depth=8,
                        max_samples=0.6,
                        max_features=0.7)

training_error = []
test_error = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=i)
    RFR.fit(X_train,Y_train)
    Y_pred_train = RFR.predict(X_train)
    Y_pred_test  = RFR.predict(X_test)
    training_error .append(np.sqrt(mean_squared_error(Y_train,Y_pred_train)))
    test_error.append(np.sqrt(mean_squared_error(Y_test,Y_pred_test)))

print("Cross validation training_error:",np.mean(training_error).round(2))
print("Cross validation test_error:",np.mean(test_error).round(2))
print('variance:',np.mean(test_error)-np.mean(training_error).round(2))

#====================================================================
# Gradiant Boosting Regressor
#====================================================================
from sklearn.ensemble import GradientBoostingRegressor
GBR = GradientBoostingRegressor(learning_rate=0.1,
                                n_estimators=500)
training_accuracy = []
test_accuracy = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=i)
    GBR.fit(X_train,Y_train)
    Y_pred_train = GBR.predict(X_train)
    Y_pred_test  = GBR.predict(X_test)
    training_error .append(np.sqrt(mean_squared_error(Y_train,Y_pred_train)))
    test_error.append(np.sqrt(mean_squared_error(Y_test,Y_pred_test)))

print("Cross validation training_error:",np.mean(training_error).round(2))
print("Cross validation test_error:",np.mean(test_error).round(2))
print('variance:',np.mean(test_error)-np.mean(training_error).round(2))

#====================================================================
# ADABOOST Regressor
#====================================================================
from sklearn.ensemble import AdaBoostRegressor
ABR = AdaBoostRegressor(estimator=DT,
                        n_estimators=400,
                        learning_rate=2)
training_accuracy = []
test_accuracy = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=i)
    ABR .fit(X_train,Y_train)
    Y_pred_train = ABR.predict(X_train)
    Y_pred_test  = ABR.predict(X_test)
    training_error .append(np.sqrt(mean_squared_error(Y_train,Y_pred_train)))
    test_error.append(np.sqrt(mean_squared_error(Y_test,Y_pred_test)))

print("Cross validation training_error:",np.mean(training_error).round(2))
print("Cross validation test_error:",np.mean(test_error).round(2))
print('variance:',np.mean(test_error)-np.mean(training_error).round(2))
#====================================================================
# XG BOOST Regressor
#====================================================================
pip install Xgboost

from Xgboost import XGBRegressor
XGR =  XGBRegressor(gamma=40,learning_rate=0.1,
                    reg_lamda=0.5,n_estimators=400)

training_accuracy = []
test_accuracy = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=i)
    ABR .fit(X_train,Y_train)
    Y_pred_train = ABR.predict(X_train)
    Y_pred_test  = ABR.predict(X_test)
    training_error .append(np.sqrt(mean_squared_error(Y_train,Y_pred_train)))
    test_error.append(np.sqrt(mean_squared_error(Y_test,Y_pred_test)))

print("Cross validation training_error:",np.mean(training_error).round(2))
print("Cross validation test_error:",np.mean(test_error).round(2))
print('variance:',np.mean(test_error)-np.mean(training_error).round(2))
