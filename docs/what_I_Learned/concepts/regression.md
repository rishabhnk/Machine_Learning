## Concepts:

### Simple Linear Regression:

  
  
### Multi Linear Regression:

*avoid the dummy variable trap - dont duplicate a dummy var bc the model cant distinguish between the effect of 2 dummies if D2 = 1 - D1 --> awlays eliminate one of the dummy variables*

### Polynomial Linear Regresion:

### Suport Vector Machine Regression:

  *need to apply feature scaling!*
  
 - The goal of the algorithm is to find the maximum distance between the hyperplane (middle line of the street) and the outer vectors (vector = point). 
 
 - This measurement is described by the variable Epsilon. Basically, we are trying to find the maximum value of Epsilon so that it can capture all the points but the algorithm will ignore the outliers. 
 
 - It is basically like using multiple linear regressions or multiple gaussian regressions becasue we also have a measure of error for our model that accounts for expected variability.

### Decision Tree Regression:

  *use high res graphing*

  - Creates splits in data in a way that each split adds value to the information we have about the points. 
  
  - Each new box that is created is called a leaf and each split that is created is correspondeding to a split in the model's decision tree. Each leaf has an average of the points it contains. 
  
  - Each new data point goes through the decision tree logic and then lands in a terminal leaf
  
  - average of the points in that terminal leaf gets assigned to the y-value of the new data point.

### Random Forest Regression:


## Syntax:

### Simple Linear Regression:

  from sklearn.linear_model import Linear_Regression

### Multi Linear Regression:

  from sklearn.linear_model import LinearRegression  

### Polynomial Linear Regresion:

  from sklearn.preprocessing import PolynomialFeatures

### Suport Vector Machine Regression:

  from sklearn.svm import SVR

### Decision Tree Regression:

  from sklearn.tree import DecisionTreeRegressor

### Random Forest Regression:

  from sklearn.ensemble import RandomForestRegressor  




Backwards Elimation

