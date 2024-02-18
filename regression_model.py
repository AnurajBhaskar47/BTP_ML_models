# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
# from sklearn.datasets import make_regression

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('./data/Training_data - Sheet1.csv')

# Check the data
# print(dataset.head())
# print(dataset.isnull().sum())

#Get the flow_velocity
y = dataset['Flow_Vel']

#Get the parameters
x = dataset.drop(['Flow_Vel'], axis = 1)

# Get the shape of the dataset
# print(f'X: {x.shape}')
# print(x.head())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=101)

# See the train, test data
# print(f'X_train: {x_train.head()}')
# print(f'X_test: {x_test.head()}')
# print(f'Y_train: {y_train.head()}')
# print(f'Y_test: {y_test.head()}')

# see the train, test shape 
# print(f'X_train: {x_train.shape}')
# print(f'X_test: {x_test.shape}')
# print(f'Y_train: {y_train.shape}')
# print(f'Y_test: {y_test.shape}')

# LogisticRegression
# log_Model = LogisticRegression()
# log_Model.fit(x_train, y_train)

# print(f'Train_Accuracy : {lr_Model.score(x_train, y_train): 3f}')
# print(f'Test_Accuracy : {lr_Model.score(x_test, y_test): 3f}')

# Decision Tree
dt_Model = DecisionTreeRegressor()

dt_Model.fit(x_train, y_train)

print('Decision_Tree')
print(f'Train_Accuracy : {dt_Model.score(x_train, y_train): 3f}')
print(f'Test_Accuracy : {dt_Model.score(x_test, y_test): 3f}')

prediction_dt = dt_Model.predict(x)

df_dt = pd.DataFrame(prediction_dt, columns=['Decision_Tree'])

df_dt.to_csv('./data/decision_tree.csv', index=False)

# Random Forest Regression
rf_Model = RandomForestRegressor()

rf_Model.fit(x_train, y_train)

print('Random_Forest')
print(f'Train_Accuracy : {rf_Model.score(x_train, y_train): 3f}')
print(f'Test_Accuracy : {rf_Model.score(x_test, y_test): 3f}')

prediction_rf = rf_Model.predict(x)

# print(prediction_rf)
# print(prediction_rf.shape)

df_rf = pd.DataFrame(prediction_rf, columns = ['Random_Forest'])

df_rf.to_csv('./data/random_forest.csv', index=False)