## What I learned so far

Some module imports
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
...

JSON Encoding
  data = json.dumps(portfolio)
  port = json.load(data)
  
  
Comprehensions
  List
    names = [holding['name'] for holding in portfolio if holding['shares'] > 100]
  Set
    {holding['name'] for holding in portfolio if holding['shares'] > 100}
  Dict
    prices = {name:float(price) for name,price in zip(uniqueNames, prideCata)}
    

Sorting - 

  portfolio.sort(key = labmbda holding: holding['name'])

Grouping - 
```

https://medium.com/machine-learning-in-practice/cheat-sheet-of-machine-learning-and-python-and-math-cheat-sheets-a4afe4e791b6
