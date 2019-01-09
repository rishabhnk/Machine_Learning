# DrivenData: Predict Blood Donations

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
trainingDataset = pd.read_csv('BloodDonationsTrainingData.csv')
testingDataset = pd.read_csv('BloodDonationsTestingData.csv')

X_train = trainingDataset.iloc[:, [1,2,3,4]].values
y_train = trainingDataset.iloc[:, 5].values

X_test = testingDataset.iloc[:, [1,2,3,4]].values

"""
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', probability = 1, random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict_proba(X_test)
y_pred = y_pred[:, 1]

y_predLabel = classifier.predict(X_test)

"""
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_pred)
"""

# Regression Metrics - Mean absolute error

""" ... """

#Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((576, 1)).astype(int), values = X_train, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()