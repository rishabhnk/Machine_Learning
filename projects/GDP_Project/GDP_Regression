# GDP Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset 
dataset = pd.read_csv('combined_results_filtered.csv')
dataset = dataset.dropna(subset=['GDP']) # drop all rows that dont have a GDP value
dataset = dataset.dropna(axis = 'columns')
X = dataset.iloc[:, 1:22].values
y = dataset.iloc[:, -1].values.reshape(-1, 1)

# Taking care of missing data - mean strategy
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis = 1)
X = imputer.fit_transform(X)

# Splitting the dataset into Traning and Testing set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test= sc_X.transform(X_test)

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
""" IMPORTANT: FEATURE SCALE Y_TEST???"""
y_test = sc_y.fit_transform(y_test)

# Fitting the Regression Model to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'linear')
regressor.fit(X_train,y_train)

#current algorithm
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train,y_train)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
polyReg = PolynomialFeatures(degree = 100)
X_poly = polyReg.fit_transform(X_train)
linReg2 = LinearRegression()
linReg2.fit(X_poly, y_train) 

# Predicting a new result
    y_pred = regressor.predict(X_test)
    y_pred = linReg2.predict(polyReg.fit_transform(X_test))

# Evaluation Metrics
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred)

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

from sklearn.model_selection import cross_val_score
acc = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10 )
acc.mean()
acc.std()

from sklearn.model_selection import GridSearchCV
param = [
            {'n_estimators': [10, 50, 75]}
        ]
grid_search = GridSearchCV(estimator = regressor, param_grid = param, scoring = 'accuracy', cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_acc = grid_search.best_params_
best_acc = grid_search.best_score_


#Date vs GDP
plt.plot(X_test[:,0], y_test, color = 'blue')
plt.title('Date vs GDP')
plt.xlabel('Date')
plt.ylabel('GDP')
plt.show()

"""
# Visualising the Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
"""
