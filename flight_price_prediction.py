''' Flight Price Prediction '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train=pd.read_excel("D:/Studies/Course Material/flight_price_prediction/Train.xlsx")

train.columns
# price is the output

train.dtypes

train.info()

train.Duration.value_counts()

train.isnull().sum()

# since the null values are very less compared to the train data shape, we'll drop the null values
train.dropna(inplace=True)

# to reset the index after the index of the null value has been dropped
train.reset_index(drop=True,inplace=True)

train.isnull().sum()

train['Journey_day']=pd.to_datetime(train.Date_of_Journey,format="%d/%m/%Y").dt.day
# to_datetime is used to work with date and time data
# format='%d/%m/%y' says that our data is in the from day/month/year(ex-18/08/2020)
# dt.day - used to extract the day

train['Journey_month']=pd.to_datetime(train.Date_of_Journey,format="%d/%m/%Y").dt.month
# dt.month - used to extract the month

# now we can drop the date of journey column
train.drop(['Date_of_Journey'],axis=1,inplace=True)

# similarly we can extract data like minutes and hours from departure time column
train['Dep_hour']=pd.to_datetime(train.Dep_Time).dt.hour
# dt.hour - used to extract the hour

train['Dep_min']=pd.to_datetime(train.Dep_Time).dt.minute
# dt.minute - used to extract the minute

# now we can drop the departure time column
train.drop(['Dep_Time'],axis=1,inplace=True)

# now we extract the hour and minutes from arrival time and drop the arrival time column
train["Arrival_hour"]=pd.to_datetime(train.Arrival_Time).dt.hour
train["Arrival_min"]=pd.to_datetime(train.Arrival_Time).dt.minute
train.drop(['Arrival_Time'],axis=1,inplace=True)

duration=list(train['Duration'])

for i in range(len(duration)):
  if len(duration[i].split()) != 2: 
   # check if duration contains only hours or mins after it splits it based on the space between the xh and ym
    if "h" in duration[i]:
      duration[i] = duration[i] + " 0m" # if the duration is only in hours then we add 0 mins
    else:
      duration[i] = "0h " + duration[i] # if the duration is only in minutes then we add 0 hours
      
''' 
duration=list(train['Duration'])
for i in range(0,6):
  print(duration[i].split()) # it splits the data by "space( )"

for i in range(0,15):
  if len(duration[i].split()) != 2:
      print(duration[i])
      
# if the data is like xh or ym and doesn't have the other component then upon splitting it's length will be 1 
  and not 2

# then we check if the data has h in it, then we add 0m and if it has m in it then we add 0h to it

'''
duration_hours=[]
duration_mins=[]
for i in range(len(duration)):
  duration_hours.append(int(duration[i].split(sep='h')[0])) 
  # extracting the value of x from xh ym after splitting it using h
  # xh ym - split using h - (x,h ym) - [0] is x
  duration_mins.append(int(duration[i].split(sep='m')[0].split()[-1])) 
  # extracting the value of y from xh ym after splitting it using m
  # xh ym - split using m - (xh y,) - [0]split() - (xh,y) - [-1] - y

'''
hrs=[]
for i in range(0,6):
  hrs.append(int(duration[i].split(sep='h')[0])) 

duration[1].split(sep='m')
duration[1].split(sep='m')[0]
duration[1].split(sep='m')[0].split()
duration[1].split(sep='m')[0].split()[-1]

'''

# adding duration hours and mins to our dataset
train['Duration_hours']=pd.Series(duration_hours)
train['Duration_mins']=pd.Series(duration_mins)

# dropping the duration column
train.drop(['Duration'],axis=1,inplace=True)

""" Handling Categorical Data

Categorical data maybe of two types -

1. Nominal - OneHotEncoder can be used
2. Ordinal - LabelEncoder can be used

"""

train['Airline'].value_counts() # Airline is nominal i.e, just names and no order or magnitude

# plotting the airlines with price
sns.catplot(y='Price',x='Airline',data=train.sort_values('Price',ascending=False),
            kind='boxen',height=6,aspect=3)
# boxen to get the box plots 
# height,aspect are figure realted

''' we can see that the price is highest for jet airways business and remaining all are almost same '''

# one hot encoding on the airline feature which is nominal
airline=pd.get_dummies(train['Airline'])
airline.columns

train.Source.value_counts()

# plotting the source with price
sns.catplot(y='Price',x='Source',data=train.sort_values('Price',ascending=False),
            kind='boxen',height=6,aspect=3)

''' This indicates that the price are slightly higher when travelling from banglore
(it also has higher outlier values) '''

# one hot encoding on the source feature which is nominal
source_col=train[['Source']]
source=pd.get_dummies(source_col)
source.head()

train.Destination.value_counts()

# plotting the source with price
sns.catplot(y='Price',x='Destination',data=train.sort_values('Price',ascending=False),
            kind='boxen',height=6,aspect=3)

''' This indicates that the price are slightly higher when travelling to delhi
(it also has higher outlier values) '''

# one hot encoding on the destination feature which is nominal
destination_col=train[['Destination']]
destination=pd.get_dummies(destination_col)
destination.head()
destination.columns

train.Route
train.Total_Stops
# the variable route and total_stops give the same information  
# since total_stops gives us better understanding via numbers which can be unserstood by the model as well
# we'll use total_stops and drop the route feature
train.drop(['Route'],axis=1,inplace=True)

train.Additional_Info.value_counts()
# almost 80% of the data in the additional info feature says no info so we'll drop this feature

train.drop(['Additional_Info'],axis=1,inplace=True)

train['Total_Stops'].value_counts()

''' total stops is an ordinal feature since as the number of stops increases then the price will increase '''

ts={'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4}

train['Total_Stops']=train['Total_Stops'].map(ts)

# concatinating the dummy variable dataset to the train dataset
train=pd.concat([train,airline,source,destination],axis=1)

train.shape

train.head()

# dropping the original features as we have the feature engineered features
train.drop(['Airline','Source','Destination'],axis=1,inplace=True)

""" Same feature engineering steps have to be done for the test dataset as well """

test=pd.read_excel("D:/Studies/Course Material/flight_price_prediction/Test.xlsx")

test.head() # it doesn't have the price variable since it's the output variable and has to be predicted

test.isnull().sum()

test['Journey_day']=pd.to_datetime(test.Date_of_Journey,format="%d/%m/%Y").dt.day
# to_datetime is used to work with date and time data
# format='%d/%m/%y' says that our data is in the from day/month/year(ex-18/08/2020)
# dt.day - used to extract the day

test['Journey_month']=pd.to_datetime(test.Date_of_Journey,format="%d/%m/%Y").dt.month
# dt.month - used to extract the month

# now we can drop the date of journey column
test.drop(['Date_of_Journey'],axis=1,inplace=True)

# similarly we can extract data like minutes and hours from departure time column
test['Dep_hour']=pd.to_datetime(test.Dep_Time).dt.hour
# dt.hour - used to extract the hour

test['Dep_min']=pd.to_datetime(test.Dep_Time).dt.minute
# dt.minute - used to extract the minute

# now we can drop the departure time column
test.drop(['Dep_Time'],axis=1,inplace=True)

# now we extract the hour and minutes from arrival time and drop the arrival time column
test["Arrival_hour"]=pd.to_datetime(test.Arrival_Time).dt.hour
test["Arrival_min"]=pd.to_datetime(test.Arrival_Time).dt.minute
test.drop(['Arrival_Time'],axis=1,inplace=True)

duration=list(test['Duration'])

for i in range(len(duration)):
  if len(duration[i].split()) != 2: 
  # check if duration contains only hours or mins after it splits it based on the space between the xh and ym
    if "h" in duration[i]:
      duration[i] = duration[i] + " 0m" # if the duration is only in hours then we add 0 mins
    else:
      duration[i] = "0h " + duration[i] # if the duration is only in minutes then we add 0 hours

duration_hours=[]
duration_mins=[]
for i in range(len(duration)):
  duration_hours.append(int(duration[i].split(sep='h')[0])) 
  # extracting the value of x from xh ym after splitting it using h
  # xh ym - split using h - (x,h ym) - [0] is x
  duration_mins.append(int(duration[i].split(sep='m')[0].split()[-1])) 
  # extracting the value of y from xh ym after splitting it using m
  # xh ym - split using m - (xh y,) - [0]split() - (xh,y) - [-1] - y
  
# adding duration hours and mins to our dataset
test['Duration_hours']=pd.Series(duration_hours)
test['Duration_mins']=pd.Series(duration_mins)

# dropping the duration column
test.drop(['Duration'],axis=1,inplace=True)
  
airline=pd.get_dummies(test['Airline'])

source_col=test[['Source']]
source=pd.get_dummies(source_col)  

destination_col=test[['Destination']]  
destination=pd.get_dummies(destination_col)

test.drop(['Route'],axis=1,inplace=True)
  
test.drop(['Additional_Info'],axis=1,inplace=True)

ts={'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4}

test['Total_Stops']=test['Total_Stops'].map(ts)

# concatinating the dummy variable dataset to the train dataset
test=pd.concat([test,airline,source,destination],axis=1)

test.shape

test.head()

# dropping the original features as we have the feature engineered features
test.drop(['Airline','Source','Destination'],axis=1,inplace=True)

""" Feature Engineering

Finding the best features which will contribute and have good relation with the output variable.

Some of the methods to find such features are:

1. Heatmap
2. feature_importance_
3. SelectKBest

"""

train.columns

x=train.loc[:,['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour',
       'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_hours',
       'Duration_mins', 'Air Asia', 'Air India', 'GoAir', 'IndiGo',
       'Jet Airways', 'Jet Airways Business', 'Multiple carriers',
       'Multiple carriers Premium economy', 'SpiceJet', 'Trujet', 'Vistara',
       'Vistara Premium economy', 'Source_Banglore', 'Source_Chennai',
       'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Banglore', 'Destination_Cochin', 'Destination_Delhi',
       'Destination_Hyderabad', 'Destination_Kolkata',
       'Destination_New Delhi']]

y=train['Price']

# finding the correlation between the features
plt.figure(figsize=(33,33))
sns.heatmap(train.corr(),annot=True,cmap='RdYlGn')
plt.show()

# important feature using ExtraTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
selection=ExtraTreesRegressor()
selection.fit(x,y)

# plot graph of feature importances for better understanding
plt.figure(figsize=(12,8))
feat_importances=pd.Series(selection.feature_importances_,index=x.columns)
feat_importances.nlargest(20).plot(kind='bar')
plt.show()

feat_importances.sort_values(ascending=False)

''' building different models to check which model works the best '''

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_train.columns

# RandomForest Regressor
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()

# XGBoost Regressor
import xgboost
xgb=xgboost.XGBRegressor()

# Linear Regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()

# Ridge Regression
from sklearn.linear_model import Ridge
ridge=Ridge()

# Lasso Regression
from sklearn.linear_model import Lasso
lasso=Lasso()

models=[rf,xgb,lr,ridge,lasso]

from sklearn import metrics

# metrics.mean_squared_error() to get the mean squared error 

rmse_scores=[]

for i in models:
    m=i.fit(x_train,y_train)
    y_pred=m.predict(x_test)
    rmse_scores.append(np.sqrt(metrics.mean_squared_error(y_pred,y_test)))
# rmse = sqrt of mse    

rmse_scores_models=pd.DataFrame(columns=['Model Name','RMSE Score'])
rmse_scores_models['Model Name']=pd.Series(['RandomForest Regression','XGB Regression','Linear Regression',
                                         'Ridge Regression','Lasso Regression'])
rmse_scores_models['RMSE Score']=pd.Series(rmse_scores)

rmse_scores_models.head()

''' as we can see that the RMSE score is the lowest for the RandomForest Regressor
    we'll go ahead with that model '''

''' Performing hyperparameter tuning '''

from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# creating the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,
                               scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, 
                               random_state=42, n_jobs = 1)

rf_random.fit(x_train,y_train)

rf_random.best_params_

prediction = rf_random.predict(x_test)

# rmse after hyperparameter tuning
rmse_hpt_score=np.sqrt(metrics.mean_squared_error(prediction,y_test))
rmse_hpt_score

# accuracy
def accuracy(i):
  acc=(i.min()/i.max())*100
  return acc

accuracy_df=pd.DataFrame({'Actual Values':y_test,'Predicted Values':prediction})
accuracy_df.head()

accuracy=(accuracy_df.apply(accuracy,axis=1)).mean()
accuracy

''' creating a pickel file of the model '''

import os 
import pickle

os.getcwd()

file = open('random_forest_regressor.pkl','wb')
# creating and opening a file named "lasso_regressor.pkl" in "wb(write byte") mode

# dumping our model and it's parameters into the file opened above
pickle.dump(rf_random,file)

file.close() # closing the file

''' reading the stored rf_random model to predict the output for the test data'''

file = open('D:/Studies/Course Material/flight_price_prediction/random_forest_regressor.pkl','rb')
model=pickle.load(file)

x_test.shape

test.shape

x_test.columns
x_train.columns

test.columns
# the column 'Trujet' is missing from the test dataset so we'll add it

test['Trujet']=0

test=test.loc[:,['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour', 'Dep_min',
       'Arrival_hour', 'Arrival_min', 'Duration_hours', 'Duration_mins',
       'Air Asia', 'Air India', 'GoAir', 'IndiGo', 'Jet Airways',
       'Jet Airways Business', 'Multiple carriers',
       'Multiple carriers Premium economy', 'SpiceJet', 'Trujet', 'Vistara',
       'Vistara Premium economy', 'Source_Banglore', 'Source_Chennai',
       'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Banglore', 'Destination_Cochin', 'Destination_Delhi',
       'Destination_Hyderabad', 'Destination_Kolkata',
       'Destination_New Delhi']]
# re-arranging the test dataset columns in the order in which the model was trained

test_pred=model.predict(test)

test_pred
# predicted price values for the test data

