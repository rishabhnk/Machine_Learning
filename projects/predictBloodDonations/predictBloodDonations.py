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
from sklearn.svm import SVC 
classifier = SVC(kernel = 'rbf', random_state = 0, probability = 1, gamma = 0.9, C = 3.25)
classifier.fit(X_train, y_train)

# Predicting the Test set results 
#(results from the X data-set)
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

# Applying Grid Search to find the best model and best parameters
from sklearn.model_selection import GridSearchCV
parameters= [ 
                {'C': [1,2,3], 'kernel': ['linear']},
                {'C': [ 3, 3.25, 3.3,], 'kernel': ['rbf'], 'gamma': [ 0.85, 0.9, 0.95]}
            ]
grid_search = GridSearchCV(estimator = classifier, param_grid= parameters, scoring = 'accuracy', cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)

best_acc = grid_search.best_score_
best_param = grid_search.best_params_

#Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((576, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()