# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 20:41:45 2022

@author: SUBHRAJIT_DEY
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math


bikes = pd.read_csv('hour.csv')

bikes_prep = bikes.copy()
bikes_prep = bikes_prep.drop(['index','date','casual','registered'], axis = 1)

#Basic checks of missing values
print(bikes_prep.isnull().sum())

#Visual using histogram
# =============================================================================
# bikes_prep.hist(rwidth = 0.9)
# plt.tight_layout()
# =============================================================================

#Four independent non-catagorical data

# =============================================================================
# plt.subplot(221)
# plt.title('Temperature Vs Demand')
# plt.scatter(bikes_prep['temp'],bikes_prep['demand'], s = 1, c = 'g')
#   
# plt.subplot(222)
# plt.title('aTemp Vs Demand')
# plt.scatter(bikes_prep['atemp'],bikes_prep['demand'],s = 1,c = 'b')
# 
# plt.subplot(223)
# plt.title('Humidity Vs Demand')
# plt.scatter(bikes_prep['humidity'], bikes_prep['demand'], s = 1, c = 'm')
# 
# 
# plt.subplot(224)
# plt.title('Windspeed Vs Demand')
# plt.scatter(bikes_prep['windspeed'], bikes_prep['demand'],s = 1, c = 'm')
# 
# plt.tight_layout()
# =============================================================================

# =============================================================================
# #Plot the Catagorical features vs Demand
# #Create a 3X3subplot
# plt.subplot(3,3,1)
# plt.title('Average Demand per Season')
# cat_list = bikes_prep['season'].unique()
# 
# #Create average demanf per season 
# 
# cat_average = bikes_prep.groupby('season').mean()['demand']
# 
# plt.bar(cat_list,cat_average)
# 
# 
# plt.subplot(3,3,2)
# plt.title('Average Demand per Month')
# cat_list = bikes_prep['month'].unique()
# 
# #Create average demanf per season 
# 
# cat_average = bikes_prep.groupby('month').mean()['demand']
# 
# plt.bar(cat_list,cat_average)
# 
# 
# 
# plt.subplot(3,3,3)
# plt.title('Average Demand per Holiday')
# cat_list = bikes_prep['holiday'].unique()
# 
# #Create average demanf per season 
# 
# cat_average = bikes_prep.groupby('holiday').mean()['demand']
# 
# plt.bar(cat_list,cat_average)
# 
# 
# plt.subplot(3,3,4)
# plt.title('Average Demand per Weekday')
# cat_list = bikes_prep['weekday'].unique()
# 
# #Create average demanf per season 
# 
# cat_average = bikes_prep.groupby('weekday').mean()['demand']
# 
# plt.bar(cat_list,cat_average)
# 
# 
# 
# plt.subplot(3,3,5)
# plt.title('Average Demand per Year')
# cat_list = bikes_prep['year'].unique()
# 
# #Create average demanf per season 
# 
# cat_average = bikes_prep.groupby('year').mean()['demand']
# 
# plt.bar(cat_list,cat_average)
# 
# 
# 
# plt.subplot(3,3,6)
# plt.title('Average Demand per Hour')
# cat_list = bikes_prep['hour'].unique()
# 
# #Create average demanf per season 
# 
# cat_average = bikes_prep.groupby('hour').mean()['demand']
# 
# plt.bar(cat_list,cat_average)
# 
# 
# plt.subplot(3,3,7)
# plt.title('Average Demand per WorkingDay')
# cat_list = bikes_prep['workingday'].unique()
# 
# #Create average demanf per season 
# 
# cat_average = bikes_prep.groupby('workingday').mean()['demand']
# 
# plt.bar(cat_list,cat_average)
# 
# 
# plt.subplot(3,3,8)
# plt.title('Average Demand per Weather')
# cat_list = bikes_prep['weather'].unique()
# 
# #Create average demanf per season 
# 
# cat_average = bikes_prep.groupby('weather').mean()['demand']
# 
# plt.bar(cat_list,cat_average)
# 
# 
# plt.tight_layout()
# =============================================================================



#Check for outliers


print(bikes_prep['demand'].describe())

print(bikes_prep['demand'].quantile([0, 0.05, 0.1, 0.15, 0.9, 0.95, 0.99]))


#Check the assumptions

correlation = bikes_prep[['temp','atemp','humidity','windspeed','demand']].corr()

#Temp and atemp shows multicolleration

bikes_prep = bikes_prep.drop(['atemp','year','workingday','windspeed','weekday'],axis = 1)

df1 = pd.to_numeric(bikes_prep['demand'],downcast = 'float')


#Autocorrelation in demAnd
# =============================================================================
# plt.acorr(df1, maxlags = 12)
# =============================================================================


#The graph is log - normal the istribution is lognormal
# =============================================================================
# 
# df1 = bikes_prep['demand']
# df2 = np.log(df1)
# 
# plt.figure()
# df1.hist(rwidth = 0.9, bins= 20)
# plt.figure()
# df2.hist(rwidth = 0.9, bins= 20)
# =============================================================================



bikes_prep['demand'] = np.log(bikes_prep['demand'])

t_1 = bikes_prep['demand'].shift(+1).to_frame()
t_1.columns = ['t-1']

t_2 = bikes_prep['demand'].shift(+2).to_frame()
t_2.columns = ['t-2']

t_3 = bikes_prep['demand'].shift(+3).to_frame()
t_3.columns = ['t-3']

bikes_prep_lag = pd.concat([bikes_prep,t_1,t_2,t_3],axis = 1)

bikes_prep_lag = bikes_prep_lag.dropna()




#Create Dummy

# =============================================================================
# dummy_df = pd.get_dummies(bikes_prep_lag,drop_first=True)
# 
# =============================================================================

print(bikes_prep_lag.dtypes)
#for dummies to work we need the datatype category

bikes_prep_lag['season'] = bikes_prep_lag['season'].astype('category')
bikes_prep_lag['holiday'] = bikes_prep_lag['holiday'].astype('category')
bikes_prep_lag['weather'] = bikes_prep_lag['weather'].astype('category')
bikes_prep_lag['month'] = bikes_prep_lag['month'].astype('category')
bikes_prep_lag['hour'] = bikes_prep_lag['hour'].astype('category')

bikes_prep_lag = pd.get_dummies(bikes_prep_lag,drop_first=True)


#Torturing the data

from sklearn.model_selection import train_test_split

#demand is time dependent feature

Y = bikes_prep_lag[['demand']]
X = bikes_prep_lag.drop(['demand'],axis = 1)

tr_size = 0.7 * len(X)
tr_size = int(tr_size)

x_train = X.values[0:tr_size]
x_test = X.values[tr_size:len(X)]

y_train = Y.values[0:tr_size]
y_test = Y.values[tr_size:len(Y)]


#Prediction useing multiple Regression

from sklearn.linear_model import LinearRegression

std_reg = LinearRegression()
std_reg.fit(x_train,y_train)

r2_train = std_reg.score(x_train,y_train)
r2_test = std_reg.score(x_test,y_test)

y_predict = std_reg.predict(x_test)


from sklearn.metrics import mean_squared_error

rmse = math.sqrt(mean_squared_error(y_test,y_predict))

Y_test_e = []
Y_predict_e = []

for i in range(0,len(y_test)):
    Y_test_e.append(math.exp(y_test[i]))
    Y_predict_e.append(math.exp(y_predict[i]))

log_sq_sum = 0
for i in range(0,len(Y_test_e)):
    log_a = math.log(Y_test_e[i]+1)
    log_p  = math.log(Y_predict_e[i]+1)
    log_diff = (log_p - log_a)**2
    log_sq_sum = log_sq_sum + log_diff

rmsle = math.sqrt(log_sq_sum/len(y_test))


print(rmsle)



