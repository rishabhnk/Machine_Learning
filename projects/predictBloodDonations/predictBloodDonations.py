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

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict_proba(X_train)
y_pred = y_pred[:, 1]
y_predLabel = classifier.predict(X_train)

# Regression Metrics - Confusion Matrix, Accuracy score, CLassification Report
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_predLabel)

classifier.score(X_train, y_train)
from sklearn.metrics import accuracy_score
accuracy_score(y_train, y_predLabel)
    """=0.9236"""
    
from sklearn.metrics import classification_report
print(classification_report(y_train, y_predLabel))

#Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((576, 1)).astype(int), values = X_train, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

