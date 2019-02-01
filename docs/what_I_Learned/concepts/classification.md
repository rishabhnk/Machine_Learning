## Concepts

### Logistic Regression:

  from sklearn.linear_model import Logistic Regression
  
### K- Nearest Neighbors (KNN):

  from sklearn.neighbors import KNeighborsClassifier

### SVM:

  from sklearn.svm import SVC

### Naive Bayes Classifier:

  from sklearn.naive_bayes import GaussianNB

### Decision Tree Classification:

  from sklearn.tree import DecisionTreeClassifier

### Random Forest Regression:

  from sklearn.ensemble import RandomForestClassifier  

### Decision Tree Regression:

  *use high res graphing*

  - Creates splits in data in a way that each split adds value to the information we have about the points. 
  
  - Each new box that is created is called a leaf and each split that is created is correspondeding to a split in the model's decision tree. Each leaf has an average of the points it contains. 
  
  - Each new data point goes through the decision tree logic and then lands in a terminal leaf
  
  - average of the points in that terminal leaf gets assigned to the y-value of the new data point.

### Random Forest Regression:
  
  - for each new data point, make each tree predict y-val, and assign the new data point the avg of all tose y-vals.
