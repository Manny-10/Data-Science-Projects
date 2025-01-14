# -*- coding: utf-8 -*-
"""
Created on Wed Jan 1  16:56:57 2025

@author: manan
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

purchaseData = pd.read_csv('Purchase_Logistic.csv')

X = purchaseData.iloc[:, [2, 3]].values
Y = purchaseData.iloc[:, 4].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

logr = LogisticRegression()
logr.fit(X_train, Y_train)

plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='viridis')
plt.suptitle('Purchase Data')
plt.xlabel('Scaled Age')
plt.ylabel('Scaled Income')
plt.grid(1, which='both')
plt.axis('tight')
plt.show()

col = sigmoid(np.dot(X_train, logr.coef_.T) + logr.intercept_)

cf = logr.coef_
xplot = np.arange(-2.0, 2, 0.01)
yplot = -(cf[0, 0] * xplot + logr.intercept_) / cf[0, 1]

plt.figure(2)
plt.scatter(X_train[:, 0], X_train[:, 1], c=col.ravel(), cmap='viridis')
plt.plot(xplot, yplot.ravel(), 'g', label='Decision Boundary')
plt.suptitle('Logistic Regression Purchase Data')
plt.xlabel('Scaled Age')
plt.ylabel('Scaled Income')
plt.grid(True, which='both')
plt.axis('tight')
plt.show()

Y_pred = logr.predict(X_test)

cmat = confusion_matrix(Y_test, Y_pred)
print('Confusion matrix of Logistic Regression is \n', cmat, '\n')

disp = ConfusionMatrixDisplay(confusion_matrix=cmat)
disp.plot()

LRscore = accuracy_score(Y_pred, Y_test)
print('Accuracy score of Logistic Regression is', 100 * LRscore, '%\n')
