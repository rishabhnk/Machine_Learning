## Concepts

### Logistic Regression:

  Essentially a linear regression for classification. x-val --> y-pred-val --> 1 or 0 based on P()
  
### K- Nearest Neighbors (KNN):

  1. K = ?
  
  2. For a new data point, take the k nearest neighbors
  
  3. Among those k nearest neighbors, count number of data points in each category
  
  4. Assign new data point to the category which you counte the most neighors
  
  5. Classified new point = done

### SVM:

  Linear
      
    - want line w maximum margin, distance between support vectors and hyperplane = max
       
    - the model learns from the extreme/boundary cases. It learns by looing at the support vectors, the point that are close to be classified by other categories.
       
    - Because of this, SVM can be very different from other ML algorithms, and in some cases can be much better.
       
  Kernel
  
    1. mapping to a higher dimension using a mapping function and using a line or plane to seprate the data, then project back to the original dimension classified groups.
    
    OR
    
    2. Kernel Trick (in same dimension) - complex decision boundary by applying kernel f(x) 

### Naive Bayes Classifier:

  Finds P(A|X) and P(B|X) using Bayes Theorem and assigns new point with the features X to the the Category (A or B) that had the higher probability. Bascically, given the features of a new data point, is it more likely to be classified as A or B? 
  
  an assumption is that the features are independent which is often not th case, but the algorithm is still applied. This is why it is called "Naive"

### Decision Tree Classification:

    *dont need feature scaling*
    
  - make splits in data that separates the different categories. each new data point goes through decision tree and lands in a terminal leaf and gets assigned to the corresponding category of that leaf


### Random Forest Regression:

 *need feature scaling*

  1.  pick k random points for a training set and build decision tree based on them
  
  2. choose number of trees
  
  3. new data pont gets assigned to the majority vote among all the trees prediction of category.
  
### Evaluation metrics:
