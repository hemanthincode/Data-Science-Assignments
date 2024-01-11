# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 16:01:00 2023

@author: bommi
"""
#--------------------------Startups data--------------------------


import numpy as np
import pandas as pd

df = pd.read_csv('50_Startups.csv')
df

df.shape
df.dtypes
list(df)

df.head()

X = df[['R&D Spend','Administration','Marketing Spend']]
Y = df['Profit']

#------------------------STANDARDSCALER---------------------------

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss_x = ss.fit_transform(X)
ss_x
ss_x = pd.DataFrame(ss_x)
ss_x
list(X)
ss_x.columns = list(X)
ss_x.head()

###-----------------------MINMAXSCALER--------------------------------

from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
mm_x = ss.fit_transform(X)
mm_x
mm_x = pd.DataFrame(ss_x)
mm_x
list(X)
mm_x.columns = list(X)
mm_x.head()

#####------------------Label encoding-------------------------------

df.dtypes
  
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
df['State']=LE.fit_transform(df['State'])


#============Multi Linear Regression using standardization==================

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(ss_x, Y)
ss_Y_pred = LR.predict(ss_x)
ss_Y_pred

from sklearn.metrics import r2_score
print("\n\t**Multi Linear Regression using min-max scalar r^2score**")
print("R^square", r2_score(Y,ss_Y_pred).round(2))


#============Multi Linear Regression using min-max scalar==================
from sklearn.preprocessing import MinMaxScaler
mm=MinMaxScaler()
mm.fit(mm_x,Y)
mm_Y_pred=LR.predict(mm_x)


#Metrics
from sklearn.metrics import r2_score
print("\n\t**Multi Linear Regression using min-max scalar r^2score**")
print("R^square", r2_score(Y,mm_Y_pred).round(2))

#============================================================================
#--------------------------Toyota Corolla--------------------------

import numpy as np
import pandas as pd

df = pd.read_csv('ToyotaCorolla.csv',encoding="latin1")
df

df.shape
df.dtypes
list(df)

df.head()

X = df[['Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight']]
Y = df['Price']

#------------------------STANDARDSCALER---------------------------

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss_x = ss.fit_transform(X)
ss_x
ss_x = pd.DataFrame(ss_x)
ss_x
list(X)
ss_x.columns = list(X)
ss_x.head()

#-----------------------MINMAXSCALER--------------------------------

from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
mm_x = ss.fit_transform(X)
mm_x
mm_x = pd.DataFrame(ss_x)
mm_x
list(X)
mm_x.columns = list(X)
mm_x.head()

#============Multi Linear Regression using standardization==================

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(ss_x, Y)
ss_Y_pred = LR.predict(ss_x)
ss_Y_pred

from sklearn.metrics import r2_score
print("\n\t**Multi Linear Regression using min-max scalar r^2score**")
print("R^square", r2_score(Y,ss_Y_pred).round(2))


#============Multi Linear Regression using min-max scalar==================
from sklearn.preprocessing import MinMaxScaler
mm=MinMaxScaler()
mm.fit(mm_x,Y)
mm_Y_pred = LR.predict(mm_x)


#Metrics
from sklearn.metrics import r2_score
print("\n\t**Multi Linear Regression using min-max scalar r^2score**")
print("R^square", r2_score(Y,mm_Y_pred).round(2))
