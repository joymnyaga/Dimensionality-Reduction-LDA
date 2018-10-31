# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: joy
"""

#Import packages
import numpy as np
import pandas as pd

#Load dataset
data=pd.read_csv("file:///D:/Data Sets - R/Pima/pima-indians-diabetes.csv")

#List of column names
list(data)

#Types of data columns
data.dtypes

#Sample of data
data.head(10)

#Find missing values
data.isnull().sum() #None

#Assign variables
X=data.iloc[:,0:8].values
y=data.iloc[:,8].values

#Split data into train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Linear Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=1)
X_train=lda.fit_transform(X_train,y_train)
X_test=lda.transform(X_test)

#Linear Regression
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(X_train,y_train)

#Predict values for cv data
pred=model.predict(X_test)

#Evaluate accuracy of model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
accuracy_score(y_test,pred)
confusion_matrix(y_test,pred)
