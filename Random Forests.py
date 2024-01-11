# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 10:14:45 2023

@author: bommi
"""
#-----------------------------------COMPANY DATA-----------------------------------------------
                              # Import the file
import numpy as np
import pandas as pd
df=pd.read_csv("Company_Data.csv")
df.head()
df.shape
df
list(df)
df.dtypes
                        # Data Transformation
         #converting the sales continous data into categorical data
df['Sales'].mean()

df['Sales'] = np.select(
    [df['Sales']<df['Sales'].mean(), 
    df['Sales']>df['Sales'].mean()],
    ['low_sales', 'high_sales'],
    default='Other')

df.head()
list(df)
                       #label encoding
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['ShelveLoc'] = LE.fit_transform(df['ShelveLoc'])
df['Urban'] = LE.fit_transform(df['Urban'])
df['US'] = LE.fit_transform(df['US'])
df['Sales'] = LE.fit_transform(df['Sales'])
df

                    #splitting as X nd Y
Y=df["Sales"]
X=df.iloc[:,1:10]

                   # standardization
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(X)
SS_X = pd.DataFrame(SS_X)
SS_X

df

                     # Data partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y,test_size=0.30)

                 # RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=100,max_depth=8,
                        max_samples=0.6,
                        max_features=0.7)

from sklearn.metrics import accuracy_score

RFC.fit(X_train,Y_train)

Y_pred_train = RFC.predict(X_train)
Y_pred_test  = RFC.predict(X_test)

training_accuracy = []
test_accuracy = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y,test_size=0.30,random_state=i)
    RFC.fit(X_train,Y_train)
    Y_pred_train = RFC.predict(X_train)
    Y_pred_test  = RFC.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))
    
k1 = pd.DataFrame(training_accuracy)
k2 = pd.DataFrame(test_accuracy)

print("\n\n")
print("\t\t\t\t\t\t\tCOMPANY_DATA\t")
print("\t\t******cross validation using RandomForestRegressor results******")
print("Cross validation training accuracy results:",k1.mean().round(2))
print("Cross validation test accuracy results:",k2.mean().round(2))
print("variance:",np.mean(k1.mean()-k2.mean()).round(2))

#==================================BOOSTING ON CAMPANY DATA===============================#

#--------------------Gradient Boosting---------------------------------------------------

from sklearn.ensemble import GradientBoostingClassifier
GBC = GradientBoostingClassifier(learning_rate=0.01,n_estimators=500)

training_accuracy = []
test_accuracy = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y,test_size=0.30,random_state=i)
    GBC.fit(X_train,Y_train)
    Y_pred_train = GBC.predict(X_train)
    Y_pred_test  = GBC.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))
    
k1 = pd.DataFrame(training_accuracy)
k2 = pd.DataFrame(test_accuracy)

print("\n\n")
print("\t\t******Gradient Boosting results******")
print("Cross validation training accuracy results:",k1.mean().round(2))
print("Cross validation test accuracy results:",k2.mean().round(2))
print("variance:",np.mean(k1.mean()-k2.mean()).round(2))

#---------------------Ada-Boosting--------------------------------------------------------

from sklearn.ensemble import AdaBoostClassifier
ABC = AdaBoostClassifier(estimator=RFC,learning_rate=2,n_estimators=200)


training_accuracy = []
test_accuracy = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y,test_size=0.30,random_state=i)
    ABC.fit(X_train,Y_train)
    Y_pred_train = ABC.predict(X_train)
    Y_pred_test  = ABC.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))
    
k1 = pd.DataFrame(training_accuracy)
k2 = pd.DataFrame(test_accuracy)

print("\n\n")
print("\t\t******Ada-Boosting results******")
print("Cross validation training accuracy results:",k1.mean().round(2))
print("Cross validation test accuracy results:",k2.mean().round(2))
print("variance:",np.mean(k1.mean()-k2.mean()).round(2))

#----------------------Extreme boosting---------------------------------------------------


pip install Xgboost

from xgboost import XGBClassifier
XGBC = XGBClassifier(gamma=40,learning_rate=0.1,reg_lamda=0.5,n_estimators=400)


training_accuracy = []
test_accuracy = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y,test_size=0.30,random_state=i)
    XGBC.fit(X_train,Y_train)
    Y_pred_train = XGBC.predict(X_train)
    Y_pred_test  = XGBC.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))
    
k1 = pd.DataFrame(training_accuracy)
k2 = pd.DataFrame(test_accuracy)

print("\n\n")
print("\t\t******Extreme boosting results******")
print("Cross validation training accuracy results:",k1.mean().round(2))
print("Cross validation test accuracy results:",k2.mean().round(2))
print("variance:",np.mean(k1.mean()-k2.mean()).round(2))

#===========================================================================================
#-------------------------------Fraud_check data------------------------------------------

                          # Import the file
import numpy as np
import pandas as pd
df=pd.read_csv("Fraud_check.csv")
df.head()
df.shape
df
list(df)
df.dtypes

# Defining Taxable.Income as Risky nd good
# Define the ranges and corresponding categories
low_range = (0, 30000)
high_range = (30001, float('inf'))

# Create a new categorical variable based on the defined ranges
df['Income_category'] = np.select(
    [df['Taxable.Income'].between(*low_range), df['Taxable.Income'].between(*high_range)],
    ['Risky', 'Good'],
    default='Other'
)

df.head()
list(df)

                   # Data Transformation
                   #label encoding
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['Undergrad'] = LE.fit_transform(df['Undergrad'])
df['Marital.Status'] = LE.fit_transform(df['Marital.Status'])
df['Urban'] = LE.fit_transform(df['Urban'])
df['Income_category'] = LE.fit_transform(df['Income_category'])

                  #splitting as X nd Y
Y=df["Income_category"]
X=df.iloc[:,[3,4]]

                   # standardization
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(X)
SS_X = pd.DataFrame(SS_X)
SS_X

                    # Data partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y,test_size=0.30)

              # RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=100,max_depth=8,
                        max_samples=0.6,
                        max_features=0.7)

from sklearn.metrics import accuracy_score

RFC.fit(X_train,Y_train)

Y_pred_train = RFC.predict(X_train)
Y_pred_test  = RFC.predict(X_test)

training_accuracy = []
test_accuracy = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y,test_size=0.30,random_state=i)
    RFC.fit(X_train,Y_train)
    Y_pred_train = RFC.predict(X_train)
    Y_pred_test  = RFC.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))
    
k1 = pd.DataFrame(training_accuracy)
k2 = pd.DataFrame(test_accuracy)

print("\n\n")
print("\t\t\t\t\t\t\tFRAUD_CHECK DATA\t")
print("\t\t******cross validation using RandomForestRegressor results******")
print("Cross validation training accuracy results:",k1.mean().round(2))
print("Cross validation test accuracy results:",k2.mean().round(2))
print("variance:",np.mean(k1.mean()-k2.mean()).round(2))

#==================================BOOSTING ON FRAUD DATA===============================#

#--------------------Gradient Boosting--------------------------------------------------

from sklearn.ensemble import GradientBoostingClassifier
GBC = GradientBoostingClassifier(learning_rate=0.01,n_estimators=500)

training_accuracy = []
test_accuracy = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y,test_size=0.30,random_state=i)
    GBC.fit(X_train,Y_train)
    Y_pred_train = GBC.predict(X_train)
    Y_pred_test  = GBC.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))
    
k1 = pd.DataFrame(training_accuracy)
k2 = pd.DataFrame(test_accuracy)

print("\n\n")
print("\t\t******Gradient Boosting results******")
print("Cross validation training accuracy results:",k1.mean().round(2))
print("Cross validation test accuracy results:",k2.mean().round(2))
print("variance:",np.mean(k1.mean()-k2.mean()).round(2))


#---------------------Ada-Boosting------------------------------------------------------

from sklearn.ensemble import AdaBoostClassifier
ABC = AdaBoostClassifier(estimator=RFC,learning_rate=2,n_estimators=200)


training_accuracy = []
test_accuracy = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y,test_size=0.30,random_state=i)
    ABC.fit(X_train,Y_train)
    Y_pred_train = ABC.predict(X_train)
    Y_pred_test  = ABC.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))
    
k1 = pd.DataFrame(training_accuracy)
k2 = pd.DataFrame(test_accuracy)

print("\n\n")
print("\t\t******Ada-Boosting results******")
print("Cross validation training accuracy results:",k1.mean().round(2))
print("Cross validation test accuracy results:",k2.mean().round(2))
print("variance:",np.mean(k1.mean()-k2.mean()).round(2))

#----------------------Extreme boosting-------------------------------------------------

pip install Xgboost

from xgboost import XGBClassifier
XGBC = XGBClassifier(gamma=40,learning_rate=0.1,reg_lamda=0.5,n_estimators=400)


training_accuracy = []
test_accuracy = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y,test_size=0.30,random_state=i)
    XGBC.fit(X_train,Y_train)
    Y_pred_train = XGBC.predict(X_train)
    Y_pred_test  = XGBC.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))
    
k1 = pd.DataFrame(training_accuracy)
k2 = pd.DataFrame(test_accuracy)

print("\n\n")
print("\t\t******Extreme boosting results******")
print("Cross validation training accuracy results:",k1.mean().round(2))
print("Cross validation test accuracy results:",k2.mean().round(2))
print("variance:",np.mean(k1.mean()-k2.mean()).round(2))