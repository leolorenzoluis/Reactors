
# Manipulating and Cleaning Data


## Exploring `DataFrame` information

&gt; **Learning goal:** By the end of this subsection, you should be comfortable finding general information about the data stored in pandas DataFrames.



```python
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
iris_df = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
```

### `DataFrame.info`
**Dataset Alert**: Iris Data about Flowers


```python
iris_df.info()
```

### `DataFrame.head`


```python
iris_df.head()
```

### Exercise:

By default, `DataFrame.head` returns the first five rows of a `DataFrame`. In the code cell below, can you figure out how to get it to show more?


```python
# Hint: Consult the documentation by using iris_df.head?

```

### `DataFrame.tail`


```python
iris_df.tail()
```



&gt; **Takeaway:** Even just by looking at the metadata about the information in a DataFrame or the first and last few values in one, you can get an immediate idea about the size, shape, and content of the data you are dealing with.

## Dealing with missing data

&gt; **Learning goal:** By the end of this subsection, you should know how to replace or remove null values from DataFrames.

**None vs NaN**

### `None`: non-float missing data


```python
import numpy as np

example1 = np.array([2, None, 6, 8])
example1
```

**Think, Pair, Share**


```python
example1.sum()
```

**Key takeaway**: Addition (and other operations) between integers and `None` values is undefined, which can limit what you can do with datasets that contain them.

### `NaN`: missing float values



```python
np.nan + 1
```


```python
np.nan * 0
```

**Think, Pair, Share**


```python
example2 = np.array([2, np.nan, 6, 8]) 
example2.sum(), example2.min(), example2.max()
```

### Exercise:


```python
# What happens if you add np.nan and None together?

```

### `NaN` and `None`: null values in pandas


```python
int_series = pd.Series([1, 2, 3], dtype=int)
int_series
```

### Exercise:


```python
# Now set an element of int_series equal to None.
# How does that element show up in the Series?
# What is the dtype of the Series?

```

### Detecting null values
`isnull()` and `notnull()`


```python
example3 = pd.Series([0, np.nan, '', None])
```


```python
example3.isnull()
```

### Exercise:


```python
# Try running example3[example3.notnull()].
# Before you do so, what do you expect to see?

```

**Key takeaway**: Both the `isnull()` and `notnull()` methods produce similar results when you use them in `DataFrame`s: they show the results and the index of those results, which will help you enormously as you wrestle with your data.

### Dropping null values


```python
example3 = example3.dropna()
example3
```


```python
example4 = pd.DataFrame([[1,      np.nan, 7], 
                         [2,      5,      8], 
                         [np.nan, 6,      9]])
example4
```

**Think, Pair, Share**


```python
example4.dropna()
```

**Drop from Columns**


```python
example4.dropna(axis='1')
```

`how='all'` will drop only rows or columns that contain all null values.

**Tip**: run `example4.dropna?`


```python
example4[3] = np.nan
example4
```

### Exercise:


```python
# How might you go about dropping just column 3?
# Hint: remember that you will need to supply both the axis parameter and the how parameter.

```

The `thresh` parameter gives you finer-grained control: you set the number of *non-null* values that a row or column needs to have in order to be kept.

**Think, Pair, Share**


```python
example4.dropna(axis='rows', thresh=3)
```

### Filling null values


```python
example5 = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'))
example5
```


```python
example5.fillna(0)
```

### Exercise:


```python
# What happens if you try to fill null values with a string, like ''?

```

**Forward-fill**


```python
example5.fillna(method='ffill')
```

**Back-fill**


```python
example5.fillna(method='bfill')
```

**Specify Axis**


```python
example4
```


```python
example4.fillna(method='ffill', axis=1)
```

### Exercise:


```python
# What output does example4.fillna(method='bfill', axis=1) produce?
# What about example4.fillna(method='ffill') or example4.fillna(method='bfill')?
# Can you think of a longer code snippet to write that can fill all of the null values in example4?

```

**Fill with Logical Data**


```python
example4.fillna(example4.mean())
```



&gt; **Takeaway:** There are multiple ways to deal with missing values in your datasets. The specific strategy you use (removing them, replacing them, or even how you replace them) should be dictated by the particulars of that data. You will develop a better sense of how to deal with missing values the more you handle and interact with datasets.

## Removing duplicate data

&gt; **Learning goal:** By the end of this subsection, you should be comfortable identifying and removing duplicate values from DataFrames.


### Identifying duplicates: `duplicated`


```python
example6 = pd.DataFrame({'letters': ['A','B'] * 2 + ['B'],
                         'numbers': [1, 2, 1, 3, 3]})
example6
```


```python
example6.duplicated()
```

### Dropping duplicates: `drop_duplicates`


```python
example6.drop_duplicates()
```


```python
example6.drop_duplicates(['letters'])
```

&gt; **Takeaway:** Removing duplicate data is an essential part of almost every data-science project. Duplicate data can change the results of your analyses and give you spurious results!

## Combining datasets: merge and join

&gt; **Learning goal:** By the end of this subsection, you should have a general knowledge of the various ways to combine `DataFrame`s.

### Categories of joins

`merge` carries out several types of joins: *one-to-one*, *many-to-one*, and *many-to-many*.

#### One-to-one joins

Consider combining two `DataFrame`s that contain different information on the same employees in a company:


```python
df1 = pd.DataFrame({'employee': ['Gary', 'Stu', 'Mary', 'Sue'],
                    'group': ['Accounting', 'Marketing', 'Marketing', 'HR']})
df1
```


```python
df2 = pd.DataFrame({'employee': ['Mary', 'Stu', 'Gary', 'Sue'],
                    'hire_date': [2008, 2012, 2017, 2018]})
df2
```

Combine this information into a single `DataFrame` using the `merge` function:


```python
df3 = pd.merge(df1, df2)
df3
```

#### Many-to-one joins


```python
df4 = pd.DataFrame({'group': ['Accounting', 'Marketing', 'HR'],
                    'supervisor': ['Carlos', 'Giada', 'Stephanie']})
df4
```


```python
pd.merge(df3, df4)
```

**Specify Key**


```python
pd.merge(df3, df4, on='group')
```

#### Many-to-many joins


```python
df5 = pd.DataFrame({'group': ['Accounting', 'Accounting',
                              'Marketing', 'Marketing', 'HR', 'HR'],
                    'core_skills': ['math', 'spreadsheets', 'writing', 'communication',
                               'spreadsheets', 'organization']})
df5
```


```python
pd.merge(df1, df5, on='group')
```

#### `left_on` and `right_on` keywords


```python
df6 = pd.DataFrame({'name': ['Gary', 'Stu', 'Mary', 'Sue'],
                    'salary': [70000, 80000, 120000, 90000]})
df6
```


```python
pd.merge(df1, df6, left_on="employee", right_on="name")
```

### Exercise:


```python
# Using the documentation, can you figure out how to use .drop() to get rid of the 'name' column?
# Hint: You will need to supply two parameters to .drop()

```

#### `left_index` and `right_index` keywords


```python
df1a = df1.set_index('employee')
df1a
```


```python
df2a = df2.set_index('employee')
df2a
```


```python
pd.merge(df1a, df2a, left_index=True, right_index=True)
```

### Exercise:


```python
# What happens if you specify only left_index or right_index?

```

**`join` for `DataFrame`s**


```python
df1a.join(df2a)
```

**Mix and Match**: `left_index`/`right_index` with `right_on`/`left_on`


```python
pd.merge(df1a, df6, left_index=True, right_on='name')
```

#### Set arithmetic for joins


```python
df5 = pd.DataFrame({'group': ['Engineering', 'Marketing', 'Sales'],
                    'core_skills': ['math', 'writing', 'communication']})
df5
```


```python
pd.merge(df1, df5, on='group')
```

**`intersection` for merge**


```python
pd.merge(df1, df5, on='group', how='inner')
```

### Exercise:


```python
# The keyword for perfoming an outer join is how='outer'. How would you perform it?
# What do you expect the output of an outer join of df1 and df5 to be?

```

**Share**


```python
pd.merge(df1, df5, how='left')
```

### Exercise:


```python
# Now run the right merge between df1 and df5.
# What do you expect to see?

```

#### `suffixes` keyword: dealing with conflicting column names


```python
df7 = pd.DataFrame({'name': ['Gary', 'Stu', 'Mary', 'Sue'],
                    'rank': [1, 2, 3, 4]})
df7
```


```python
df8 = pd.DataFrame({'name': ['Gary', 'Stu', 'Mary', 'Sue'],
                    'rank': [3, 1, 4, 2]})
df8
```


```python
pd.merge(df7, df8, on='name')
```

**Using `_` to merge same column names**


```python
pd.merge(df7, df8, on='name', suffixes=['_left', '_right'])
```

### Concatenation in NumPy
**One-dimensional arrays**


```python
x = [1, 2, 3]
y = [4, 5, 6]
z = [7, 8, 9]
np.concatenate([x, y, z])
```

**Two-dimensional arrays**


```python
x = [[1, 2],
     [3, 4]]
np.concatenate([x, x], axis=1)
```

### Concatenation in pandas

**Series**


```python
ser1 = pd.Series(['a', 'b', 'c'], index=[1, 2, 3])
ser2 = pd.Series(['d', 'e', 'f'], index=[4, 5, 6])
pd.concat([ser1, ser2])
```

**DataFrames**


```python
df9 = pd.DataFrame({'A': ['a', 'c'],
                    'B': ['b', 'd']})
df9
```


```python
pd.concat([df9, df9])
```

**Re-indexing**


```python
pd.concat([df9, df9], ignore_index=True)
```

**Changing Axis**


```python
pd.concat([df9, df9], axis=1)
```

&gt; Note that while pandas will display this without error, you will get an error message if you try to assign this result as a new `DataFrame`. Column names in `DataFrame`s must be unique.

### Concatenation with joins


```python
df10 = pd.DataFrame({'A': ['a', 'd'],
                     'B': ['b', 'e'],
                     'C': ['c', 'f']})
df10
```


```python
df11 = pd.DataFrame({'B': ['u', 'x'],
                     'C': ['v', 'y'],
                     'D': ['w', 'z']})
df11
```


```python
pd.concat([df10, df11])
```


```python
pd.concat([df10, df11], join='inner')
```


```python
pd.concat([df10, df11], join_axes=[df10.columns])
```

#### `append()`


```python
df9.append(df9)
```

**Important point**: Unlike the `append()` and `extend()` methods of Python lists, the `append()` method in pandas does not modify the original object. It instead creates a new object with the combined data.

&gt; **Takeaway:** A large part of the value you can provide as a data scientist comes from connecting multiple, often disparate datasets to find new insights. Learning how to join and merge data is thus an essential part of your skill set.
