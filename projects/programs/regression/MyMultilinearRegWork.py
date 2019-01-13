# MultiLinREg
"""
Created on Fri Dec 28 23:24:51 2018

@author: risha
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#categorical enocoding
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap
X = X[:, 1:] # Removing 1 column from X, some libraries do this automatically


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regresion to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)

#Building the optimal model using Backward Elimination
    import statsmodels.formula.api as sm #eval p-vals and eval stats significance of features
   
    X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1) # adding matrix X to a column of ones
    # so there is an x0=1 column as the first column, collowed by x1, x2... this allows model to take into account the b0 constant
    
    X_opt = X[:,[0,1,2,3,4,5]]""" same as X=X?????  """
    regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #OLS (a class) = ordinary least squares
    regressor_OLS.summary() # comp output = p-value
    
    X_opt = X[:,[0,1,3,4,5]]
    regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() 
    regressor_OLS.summary() 
    
    X_opt = X[:,[0,3,4,5]]
    regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() 
    regressor_OLS.summary()
    
    X_opt = X[:,[0,3,5]]
    regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() 
    regressor_OLS.summary()

    X_opt = X[:,[0,3]]
    regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() 
    regressor_OLS.summary()




