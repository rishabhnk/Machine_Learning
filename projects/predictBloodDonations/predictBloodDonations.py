# DrivenData: Predict Blood Donations

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
trainingDataset = pd.read_csv('BloodDonationsTrainingData.csv')
testingDataset = pd.read_csv('BloodDonationsTestingData.csv')

X = trainingDataset.iloc[:, [1,2,3,4]].values
y = trainingDataset.iloc[:, 5].values

X_testingDataset = testingDataset.iloc[:, [1,2,3,4]].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

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
y_pred = classifier.predict_proba(X_test)
y_pred = y_pred[:, 1]
y_predLabel = classifier.predict(X_test)

# Regression Metrics - Confusion Matrix, Accuracy score, CLassification Report, k-fold cross validation
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predLabel)

classifier.score(X_test, y_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predLabel)
    
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predLabel))

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

#Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((576, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

