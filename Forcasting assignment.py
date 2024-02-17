# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:18:09 2024

@author: bommi
"""

import numpy as np
import pandas as pd
pd.set_option('display.max_columns',20)

# Read the Excel file
Airlines = pd.read_excel("Airlines+Data.xlsx")
Airlines.shape
list(Airlines)
Airlines.Passengers.plot()

Airlines.head()

Airlines.Month

# Preprocess the data
Airlines['Date'] = pd.to_datetime(Airlines['Month'], format="%b-%y")
Airlines['month'] = Airlines['Date'].dt.month_name()
Airlines['year'] = Airlines['Date'].dt.year

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

# Boxplot for ever
plt.figure(figsize=(8,6))
sns.boxplot(x="month",y="Passengers",data=Airlines)

plt.figure(figsize=(8,6))
sns.boxplot(x="year",y="Passengers",data=Airlines)

plt.figure(figsize=(12,3))
sns.lineplot(x="year",y="Passengers",data=Airlines)

# Splitting data
Airlines.shape
Train = Airlines.head(96)
Test = Airlines.tail(5)
Test


# Preprocess the data
Airlines['Date'] = pd.to_datetime(Airlines['Month'], format="%b-%y")
Airlines['month'] = Airlines['Date'].dt.month_name()
Airlines['year'] = Airlines['Date'].dt.year

# Heatmap
plt.figure(figsize=(12,8))
heatmap_y_month = pd.pivot_table(data=Airlines, values="Passengers", index="year", columns="month", fill_value=0)
sns.heatmap(heatmap_y_month, annot=True, fmt="g")

# Boxplot
plt.figure(figsize=(8,6))
sns.boxplot(x="month", y="Passengers", data=Airlines)

# Linear model
linear_model = smf.ols('Passengers ~ Date', data=Airlines).fit()
pred_linear = pd.Series(linear_model.predict(pd.DataFrame(Airlines['Date'])))
rmse_linear = np.sqrt(np.mean((np.array(Airlines['Passengers']) - np.array(pred_linear))**2))
print("RMSE Linear:", rmse_linear)

# Exponential
Exp = smf.ols('np.log(Passengers) ~ Date', data=Airlines).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Airlines['Date'])))
rmse_Exp = np.sqrt(np.mean((np.array(Airlines['Passengers']) - np.array(np.exp(pred_Exp)))**2))
print("RMSE Exponential:", rmse_Exp)

# Additive seasonality
add_sea = smf.ols('Passengers ~ month', data=Airlines).fit()
pred_add_sea = pd.Series(add_sea.predict(pd.DataFrame(Airlines['month'])))
rmse_add_sea = np.sqrt(np.mean((np.array(Airlines['Passengers']) - np.array(pred_add_sea))**2))
print("RMSE Additive Seasonality:", rmse_add_sea)

# Multiplicative Seasonality
Mul_sea = smf.ols('np.log(Passengers) ~ month', data=Airlines).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(pd.DataFrame(Airlines['month']))).reset_index(drop=True)
rmse_Mult_sea = np.sqrt(np.mean((np.array(Airlines['Passengers']) - np.array(np.exp(pred_Mult_sea)))**2))
print("RMSE Multiplicative Seasonality:", rmse_Mult_sea)

# Multiplicative Additive Seasonality
Mul_Add_sea = smf.ols('np.log(Passengers) ~ Date + month', data=Airlines).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(pd.DataFrame(Airlines[['Date', 'month']]))).reset_index(drop=True)
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Airlines['Passengers']) - np.array(np.exp(pred_Mult_add_sea)))**2))
print("RMSE Multiplicative Additive Seasonality:", rmse_Mult_add_sea)

# Compare the results
data = {
    "MODEL": pd.Series(["rmse_linear", "rmse_Exp", "rmse_add_sea", "rmse_Mult_sea", "rmse_Mult_add_sea"]),
    "RMSE_Values": pd.Series([rmse_linear, rmse_Exp, rmse_add_sea, rmse_Mult_sea, rmse_Mult_add_sea])
}
table_rmse = pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])

# Visualize the predictions
plt.plot(Airlines['Date'], Airlines['Passengers'], label='Actual')
plt.plot(Airlines['Date'], pred_Mult_add_sea, label='Multiplicative Additive Seasonality Prediction')
plt.xlabel('Date')
plt.ylabel('Passengers')
plt.title('Multiplicative Additive Seasonality Model Prediction')
plt.legend()
plt.show()


#=======================================================================================
#--------------------------------------CocaCola_Sales_Rawdata---------------------------------
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# Read the Excel file
CocaCola = pd.read_excel("CocaCola_Sales_Rawdata.xlsx")

# Preprocess the data (if needed)
# For example, converting date columns to datetime format, extracting month/year, etc.

# Linear model
linear_model = smf.ols('Sales ~ Quarter', data=CocaCola).fit()
pred_linear = linear_model.predict(CocaCola[['Quarter']])
rmse_linear = np.sqrt(np.mean((CocaCola['Sales'] - pred_linear)**2))
print("RMSE Linear:", rmse_linear)

# Exponential model
Exp = smf.ols('np.log(Sales) ~ Quarter', data=CocaCola).fit()
pred_Exp = np.exp(Exp.predict(CocaCola[['Quarter']]))
rmse_Exp = np.sqrt(np.mean((CocaCola['Sales'] - pred_Exp)**2))
print("RMSE Exponential:", rmse_Exp)

# Additive seasonality model
add_sea = smf.ols('Sales ~ Quarter', data=CocaCola).fit()
pred_add_sea = add_sea.predict(CocaCola[['Quarter']])
rmse_add_sea = np.sqrt(np.mean((CocaCola['Sales'] - pred_add_sea)**2))
print("RMSE Additive Seasonality:", rmse_add_sea)

# Multiplicative seasonality model
Mul_sea = smf.ols('np.log(Sales) ~ Quarter', data=CocaCola).fit()
pred_Mult_sea = np.exp(Mul_sea.predict(CocaCola[['Quarter']]))
rmse_Mult_sea = np.sqrt(np.mean((CocaCola['Sales'] - pred_Mult_sea)**2))
print("RMSE Multiplicative Seasonality:", rmse_Mult_sea)

# Multiplicative additive seasonality model
Mul_Add_sea = smf.ols('np.log(Sales) ~ Quarter', data=CocaCola).fit()
pred_Mult_add_sea = np.exp(Mul_Add_sea.predict(CocaCola[['Quarter']]))
rmse_Mult_add_sea = np.sqrt(np.mean((CocaCola['Sales'] - pred_Mult_add_sea)**2))
print("RMSE Multiplicative Additive Seasonality:", rmse_Mult_add_sea)

# Compare the results
data = {
    "MODEL": pd.Series(["rmse_linear", "rmse_Exp", "rmse_add_sea", "rmse_Mult_sea", "rmse_Mult_add_sea"]),
    "RMSE_Values": pd.Series([rmse_linear, rmse_Exp, rmse_add_sea, rmse_Mult_sea, rmse_Mult_add_sea])
}
table_rmse = pd.DataFrame(data)
table_rmse = table_rmse.sort_values(['RMSE_Values'])
print(table_rmse)

# Visualize the predictions
plt.plot(CocaCola['Sales'], label='Actual')
plt.plot(pred_Mult_add_sea, label='Multiplicative Additive Seasonality Prediction')
# plt.plot(pred_Quad, label='Quadratic Prediction')
# plt.plot(pred_add_sea_quad, label='Additive Seasonality Quadratic Prediction')
plt.xlabel('Index')
plt.ylabel('Sales')
plt.title('Sales Prediction Models Comparison')
plt.legend()
plt.show()
