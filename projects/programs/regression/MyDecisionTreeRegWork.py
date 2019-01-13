# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting the Regression Model to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)


 # Predicting a new result
y_pred = regressor.predict(6.5) 
    # anything in the 5.5 - 6.5 range will be predicted to be 150,000

"""
# Visualising the Regression results
# DO NOT USE THIS, NOT APPROPRIATE FOR DTR BC NON-CONT MODEL
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (DTR Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
"""

# Visualising the Regression results (for higher resolution and smoother curve)
# WE NEED HIGHER RESOLTION SO WE WILL USE THIS
X_grid = np.arange(min(X), max(X), 0.0001)#The smaller the param, the higher the res
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
#flatlines = any data point that lies in that terminal leaf will get the value of the average in that leaf