## File Work
Opening:
  
Closing:

Reading:

## Data Pre-Processing
  Basics:
    Text Strings: 
      text = array of characters
      ...
## Machine Learning:

### Taking care of missing data

  =Using Imputer class to create an Imputer object that will fill the missing values with the mean of row/column. (1 = col, 0 = row) Use fit_transform on the specified indeces of the X matrix.

---------------------------------

Syntax:

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values="NaN", strategy="mean", axis = 0)

X[:,1:3] = imputer.fit_transform(X[:,1:3])

### Encoding categorical

  = Using the LabelEncoder class from the sklearn.preprocessing library to make an object that will encode all categorical variables in the specified indeces into numbers. 
  
  = But because encoding into numbers will give an inherent ordering, we can use the OneHotEncoder class from the same library to make an object that will fit and transform our categorical variables into dummy variables. (be sure to include the .toarray()) Because you specify which columns are categorical that you want to one hot encode, you can pass the whole X matrix into the fit_transform.

---------------------------------

Syntax:

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

X[:,0] = labelencoder.fit_transform(X[:,0])


from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder(categorical_features = [0])

X = onehotencoder.fit_transform(X).toarray()

### Splitting the dataset into Traning and Testing set
  
  = If you want to split the testing and training set, train_test_split from cross_validation (only preprocessing code from cross_validation)

  = Use the train_test_split class from the library sklearn.cross_validation to split the X and y into the training and testing sets.

-----------------------------------------

from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

### Feature scaling

  = If you want to feature scale, or scale the X and y to make them standard, use the StandardScalar class from the sklearn.preprocessing module

  = Use the StandardScalar class to make an object that will fit_transform the X_train and X_test matrices and another object to fit_transform y_train
  
------------------------------------------

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.fit_transform(X_test)

sc_y = StandardScaler()

y_train = sc_y.fit_transform(y_train)

