# File Work
Opening:
  
Closing:

Reading:

# Data Pre-Processing
  Basics:
    Text Strings: 
      text = array of characters
      ...
 ## Machine Learning:
  
# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis = 0)
X[:,1:3] = imputer.fit_transform(X[:,1:3])
# Encoding categorical
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:,0] = labelencoder.fit_transform(X[:,0])





