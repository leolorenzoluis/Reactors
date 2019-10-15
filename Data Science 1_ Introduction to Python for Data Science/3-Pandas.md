
# Introduction to Pandas


```python
import pandas as pd
```


```python
import numpy as np
```

## Fundamental panda data structures

### `Series` objects in pandas


```python
series_example = pd.Series([-0.5, 0.75, 1.0, -2])
series_example
```


```python
series_example.values
```


```python
series_example.index
```


```python
series_example[1]
```


```python
series_example[1:3]
```

### Explicit Indices


```python
series_example2 = pd.Series([-0.5, 0.75, 1.0, -2], index=['a', 'b', 'c', 'd'])
series_example2
```


```python
series_example2['b']
```

### Exercise:


```python
# Do explicit Series indices work *exactly* the way you might expect?
# Try slicing series_example2 using its explicit index and find out.

```

### Series vs Dictionary

**Think, Pair, Share** 


```python
population_dict = {'France': 65429495,
                   'Germany': 82408706,
                   'Russia': 143910127,
                   'Japan': 126922333}
population_dict
```


```python
population = pd.Series(population_dict)
population
```

### Interacting with Series


```python
population['Russia']
```

### Exercise


```python
# Try slicing on the population Series on your own.
# Would slicing be possible if Series keys were not ordered?
population['Germany':'Russia']
```


```python
# Try running population['Albania'] = 2937590 (or another country of your choice)
# What order do the keys appear in when you run population? Is it what you expected?

```


```python
population
```


```python
pop2
```


```python
pop2 = pd.Series({'Spain': 46432074, 'France': 102321, 'Albania': 50532})
population + pop2
```

### `DataFrame` object in pandas


```python
area_dict = {'Albania': 28748,
             'France': 643801,
             'Germany': 357386,
             'Japan': 377972,
             'Russia': 17125200}
area = pd.Series(area_dict)
area
```


```python
countries = pd.DataFrame({'Population': population, 'Area': area})
countries
```


```python
countries['Capital'] = ['Tirana', 'Paris', 'Berlin', 'Tokyo', 'Moscow']
countries
```


```python
countries = countries[['Capital', 'Area', 'Population']]
countries
```


```python
countries['Population Density'] = countries['Population'] / countries['Area']
countries
```


```python
countries['Area']
```

### Exercise


```python
# Now try accessing row data with a command like countries['Japan']

```

**Think, Pair, Share**


```python
countries.loc['Japan']
```


```python
countries.loc['Japan']['Area']
```

### Exercise


```python
# Can you think of a way to return the area of Japan without using .iloc?
# Hint: Try putting the column index first.
# Can you slice along these indices as well?

```

### DataSeries Creation


```python
countries['Debt-to-GDP Ratio'] = np.nan
countries
```


```python
debt = pd.Series([0.19, 2.36], index=['Russia', 'Japan'])
countries['Debt-to-GDP Ratio'] = debt
countries
```


```python
del countries['Capital']
countries
```


```python
countries.T
```


```python
pd.DataFrame(np.random.rand(3, 2),
             columns=['foo', 'bar'],
             index=['a', 'b', 'c'])
```

## Manipulating data in pandas

### Index objects in pandas


```python
series_example = pd.Series([-0.5, 0.75, 1.0, -2], index=['a', 'b', 'c', 'd'])
ind = series_example.index
ind
```


```python
ind[1]
```


```python
ind[::2]
```

**Share**


```python
ind[1] = 0
```

### Set Properties


```python
ind_odd = pd.Index([1, 3, 5, 7, 9])
ind_prime = pd.Index([2, 3, 5, 7, 11])
```

**Think, Pair, Share**  
In the code cell below, try out the intersection (`ind_odd &amp; ind_prime`), union (`ind_odd | ind_prime`), and the symmetric difference (`ind_odd ^ ind_prime`) of `ind_odd` and `ind_prime`.


```python

```

### Data Selection in Series


```python
series_example2 = pd.Series([-0.5, 0.75, 1.0, -2], index=['a', 'b', 'c', 'd'])
series_example2
```


```python
series_example2['b']
```


```python
'a' in series_example2
```


```python
series_example2.keys()
```


```python
list(series_example2.items())
```


```python
series_example2['e'] = 1.25
series_example2
```

### Indexers: `loc` and `iloc`

**Think, Pair, Share**


```python
series_example2.loc['a']
```


```python
series_example2.loc['a':'c']
```

**Share**


```python
series_example2.iloc[0]
```


```python
series_example2.iloc[0:2]
```

### Data Selection in DataFrames


```python
area = pd.Series({'Albania': 28748,
                  'France': 643801,
                  'Germany': 357386,
                  'Japan': 377972,
                  'Russia': 17125200})
population = pd.Series ({'Albania': 2937590,
                         'France': 65429495,
                         'Germany': 82408706,
                         'Russia': 143910127,
                         'Japan': 126922333})
countries = pd.DataFrame({'Area': area, 'Population': population})
countries
```


```python
countries['Area']
```


```python
countries['Population Density'] = countries['Population'] / countries['Area']
countries
```

### DataFrame as two-dimensional array


```python
countries.values
```


```python
countries.T
```


```python
countries.iloc[:3, :2]
```


```python
countries.loc[:'Germany', :'Population']
```

### Exercise


```python
# Can you think of how to combine masking and fancy indexing in one line?
# Your masking could be somthing like countries['Population Density'] > 200
# Your fancy indexing could be something like ['Population', 'Population Density']
# Be sure to put the the masking and fancy indexing inside the square brackets: countries.loc[]

```

# Operating on Data in Pandas

**Think, Pair, Share** For each of these Sections.

## Index alignment with Series

For our first example, suppose we are combining two different data sources and find only the top five countries by *area* and the top five countries by *population*:


```python
area = pd.Series({'Russia': 17075400, 'Canada':  9984670,
                  'USA': 9826675, 'China': 9598094, 
                  'Brazil': 8514877}, name='area')
population = pd.Series({'China': 1409517397, 'India': 1339180127,
                        'USA': 324459463, 'Indonesia': 322179605, 
                        'Brazil': 207652865}, name='population')
```


```python
# Now divide these to compute the population density
pop_density = area/population
pop_density
```


```python
series1 = pd.Series([2, 4, 6], index=[0, 1, 2])
series2 = pd.Series([3, 5, 7], index=[1, 2, 3])
series1 + series2
```


```python
series1.add(series2, fill_value=0)
```

Much better!

## Index alignment with DataFrames


```python
rng = np.random.RandomState(42)
df1 = pd.DataFrame(rng.randint(0, 20, (2, 2)),
                   columns=list('AB'))
df1
```


```python
df2 = pd.DataFrame(rng.randint(0, 10, (3, 3)),
                   columns=list('BAC'))
df2
```


```python
# Add df1 and df2. Is the output what you expected?
df1 + df2
```


```python
fill = df1.stack().mean()
df1.add(df2, fill_value=fill)
```

## Operations between DataFrames and Series

Index and column alignment gets maintained in operations between a `DataFrame` and a `Series` as well. To see this, consider a common operation in data science, wherein we find the difference of a `DataFrame` and one of its rows. Because pandas inherits ufuncs from NumPy, pandas will compute the difference row-wise by default:


```python
df3 = pd.DataFrame(rng.randint(10, size=(3, 4)), columns=list('WXYZ'))
df3
```


```python
df3 - df3.iloc[0]
```


```python
df3.subtract(df3['X'], axis=0)
```


```python
halfrow = df3.iloc[0, ::2]
halfrow
```


```python
df3 - halfrow
```
