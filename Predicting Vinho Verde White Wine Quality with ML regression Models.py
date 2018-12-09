#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 03:22:32 2018

@author: harsh
"""
##Logistic regresssion with White wine quality
import os
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

#Setting up working directories
os.chdir(r'/Users/harsh/Desktop/DM project Harsh/DM project Harsh final')

#Importing The Dataset
wq=pd.read_csv('WQW.csv',delimiter = ";")

#copy of dataset
wqc=wq.copy()

#checking basic information of data
wqc.shape
wqc.info()
desc=wqc.describe()
wqc.head()
wqc.quality.value_counts()
wqc.columns

#Checking for Missing values
plt.figure(figsize=(8,4))
sns.heatmap(wqc.isnull(),cbar=False,cmap='viridis',yticklabels=False)
plt.title('Missing value in the dataset')

#describing Feature distribution
selected_features=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol', 'quality']
n_rows = 4
n_cols = 3
fig=plt.figure()
for i, var_name in enumerate(selected_features):
    ax=fig.add_subplot(n_rows,n_cols,i+1)
    wqc[var_name].hist(bins=10,ax=ax,color='green')
    ax.set_title(var_name+" Distribution")
fig.tight_layout()
fig.set_size_inches(10,10)
plt.show()

#a pearson correlation heatmap (matrix) was plotted for correlation.
plt.figure(figsize = (9, 7))
corr =wqc.corr()
sns.heatmap(corr, cmap="RdBu",xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,annot=True)

#Separating the Independent and Dependent Variables
x=wqc[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']]
y = wqc[['quality']]


# Split into training and test set
# 80% of the input for training and 20% for testing

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = 0.2,random_state=42)
print (x_train.shape, y_train.shape)
print (x_test.shape, y_test.shape)

# Applying Logistic Regression / Training or Model Fitting
regressor = LogisticRegression()
regressor.fit(x_train, y_train)
print('Accuracy:',regressor.score(x_train,y_train))

#K-fold Cross Validation (with R2 value)
accuracies = cross_val_score(estimator = regressor, X = x_train,y = y_train, cv = 10, scoring = 'accuracy')
accuracy = accuracies.mean()
print('r2 = {}'.format(accuracy))

#Predicting the test set
y_pred = pd.DataFrame(regressor.predict(x_test))

#y_pred.columns=['Quality']
print(y_pred[0:5])

# Accuracy Score
accuracy_score(y_test,y_pred)

# Confusion Matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)


