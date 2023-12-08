# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 10:09:46 2023

@author: bommi
"""

#====================Delivery Time====================================
# Import the file

import numpy as np
import pandas as pd
df = pd.read_csv('delivery_time.csv')
df
# Exploratory Data Analysis 
import matplotlib.pyplot as plt
plt.scatter(df['Delivery Time'],df['Sorting Time'])
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.title('Linear Regression: Delivery Time vs Sorting Time')
plt.show()

# Split the data
X = df[['Sorting Time']]
Y = df['Delivery Time']
# Build a linear regression model
from sklearn.linear_model import LinearRegression
LR = LinearRegression()

LR.fit(X,Y)

LR.predict(X)
Y_predict = LR.predict(X)
# Evaluate the model
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y, Y_predict)
print('Mean Square Error',mse.round(2))
print('Root Mean Square Error',np.sqrt(mse).round(2))

# Simple linear regression tranformation techniques
print("\n\n\t\t**TRANSFORMATION TECHNIQUES**")
# X square   
df['Sorting Time_2']=df['Sorting Time']*df['Sorting Time']
df.head()

Y=df["Delivery Time"]
X=df[["Sorting Time_2"]]

from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
Y_pred=LR.predict(X)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y,Y_pred)
print("\n**X square transformation results**")
print("Mean square error",mse.round(2))
print("root mean squared error",np.sqrt(mse).round(2))


# sqrt X    
df['Sorting Time_3']=np.sqrt(df['Sorting Time'])
df.head()

Y=df["Delivery Time"]
X=df[["Sorting Time_3"]]

from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
Y_pred=LR.predict(X)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y,Y_pred)
print("\n**sqrt X transformation results**")
print("Mean square error",mse.round(2))
print("root mean squared error",np.sqrt(mse).round(2))


# log X    
df['Sorting Time_4']=np.log(df['Sorting Time'])
df.head()

Y=df["Delivery Time"]
X=df[["Sorting Time_4"]]

from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
Y_pred=LR.predict(X)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y,Y_pred)
print("\n**log X transformation results**")
print("Mean square error",mse.round(2))
print("root mean squared error",np.sqrt(mse).round(2))


# Inverse X    
df['Sorting Time_5']=(1/(df['Sorting Time']))
df.head()

Y=df["Delivery Time"]
X=df[["Sorting Time_5"]]

from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
Y_pred=LR.predict(X)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y,Y_pred)
print("\n**Inverse X transformation results**")
print("Mean square error",mse.round(2))
print("root mean squared error",np.sqrt(mse).round(2))

########################################################### ###############
#=============================Salary_Data======================================

# Import the file
import pandas as pd
df = pd.read_csv('Salary_Data.csv')
df
# EDA
import matplotlib.pyplot as plt
plt.scatter(df['YearsExperience'],df['Salary'])
plt.title('Scatter plot of Salary vs Years of Experience')
plt.show()
# Split the data
X = df[['YearsExperience']]
Y = df['Salary']
# Build a linear regression model
from sklearn.linear_model import LinearRegression
LR = LinearRegression()

LR.fit(X,Y)

LR.predict(X)
Y_predict = LR.predict(X)
# Evaluate the model
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y, Y_predict)
print('Mean Square Error',mse.round(2))
print('Root Mean Square Error',np.sqrt(mse).round(2))


# Simple linear regression tranformation techniques
print("\n\n\t\t**TRANSFORMATION TECHNIQUES**")
# X square   
df['YearsExperience_2']=df['YearsExperience']*df['YearsExperience']
df.head()

Y=df["Salary"]
X=df[["YearsExperience_2"]]

from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
Y_pred=LR.predict(X)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y,Y_pred)
print("\n**X square transformation results**")
print("Mean square error",mse.round(2))
print("root mean squared error",np.sqrt(mse).round(2))

# sqrt X    
df['YearsExperience_3']=np.sqrt(df['YearsExperience'])
df.head()

Y=df["Salary"]
X=df[["YearsExperience_3"]]

from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
Y_pred=LR.predict(X)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y,Y_pred)
print("\n**sqrt X transformation results**")
print("Mean square error",mse.round(2))
print("root mean squared error",np.sqrt(mse).round(2))

#   log X    
df['YearsExperience_4']=np.log(df['YearsExperience'])
df.head()

Y=df["Salary"]
X=df[["YearsExperience_4"]]

from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
Y_pred=LR.predict(X)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y,Y_pred)
print("\n**log X transformation results**")
print("Mean square error",mse.round(2))
print("root mean squared error",np.sqrt(mse).round(2))

#   inverse X    
df['YearsExperience_5']=(1/(df['YearsExperience']))
df.head()

Y=df["Salary"]
X=df[["YearsExperience_5"]]

from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
Y_pred=LR.predict(X)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y,Y_pred)
print("\n**Inverse X transformation results**")
print("Mean square error",mse.round(2))
print("root mean squared error",np.sqrt(mse).round(2))