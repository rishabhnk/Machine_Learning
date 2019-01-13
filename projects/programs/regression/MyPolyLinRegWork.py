#Polynomial Regresson

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #must be a matrix, 2 is not included
y = dataset.iloc[:, 2].values#= a vector

"""
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


#Fitting Linear Regresson to the dataset (t compare to polylinreg)
from sklearn.linear_model import LinearRegression
linReg = LinearRegression()
linReg.fit(X,y)

#fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polyReg = PolynomialFeatures(degree = 4) #will transform X matrix --> matrix of  X, X^2, X^3, X_n
X_poly = polyReg.fit_transform(X)# fit object to X and then transform X into X_poly

linReg2 = LinearRegression()
linReg2.fit(X_poly, y) 

#Vizualizing the lin reg results

plt.scatter(X, y, color = 'red')
plt.plot(X, linReg.predict(X), color = 'blue')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('Truth or Bluff(LinReg)')
plt.show()

# Vizualizing the polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, linReg2.predict(polyReg.fit_transform(X)), color = 'blue')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('Truth or Bluff(PolyLinReg)')
plt.show()

#predicting a new results with LinReg vs PolyLinReg
linReg.predict(6.5)

linReg2.predict(polyReg.fit_transform(6.5))

