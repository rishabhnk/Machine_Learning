## What I learned so far

Some module imports
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

...

JSON Encoding - 
  data = json.dumps(portfolio)
  port = json.load(data)
  
  
Comprehensions - 
   *names = start/ input-var for iter-var in data structure /end*
  List - 
    names = [holding['name'] for holding in portfolio if holding['shares'] > 100]
  Set - 
    {holding['name'] for holding in portfolio if holding['shares'] > 100}
  Dict - 
    prices = {name:float(price) for name,price in zip(uniqueNames, prideCata)}
    
Sorting - 
 *portfolio.sort(key = labmda input: output)*
  portfolio.sort(key = labmbda holding: holding['name'])

Grouping - 
  for name, items in itertools.groupby(portfolio, key = lambda holding: holding['name'])
  
  (group into dictionary)
  by_name = {name:list(items) for name, items in itertools.groupby(portfolio, key = lambda holding: holding['name'])} 
  
df apply - 
  x['x1x2'] = x.apply(lambda row: row['x1']*row['x2'], axis = 1)
  
df merge - 
  m = pd.merge(t1, t2, on = 'var_to_be_merged_on')
  
df cool selection - 
  x[x[0]<5] = "a df made from the x df that only consists of the rows where that condition is true"
  x[['a'], ['b']]
```

https://medium.com/machine-learning-in-practice/cheat-sheet-of-machine-learning-and-python-and-math-cheat-sheets-a4afe4e791b6
