
# Introduction to Pandas

Having explored NumPy, it is time to get to know the other workhorse of data science in Python: pandas. The pandas library in Python really does a lot to make working with data--and importing, cleaning, and organizing it--so much easier that it is hard to imagine doing data science in Python without it.

But it was not always this way. Wes McKinney developed the library out of necessity in 2008 while at AQR Capital Management in order to have a better tool for dealing with data analysis. The library has since taken off as an open-source software project that has become a mature and integral part of the data science ecosystem. (In fact, some examples in this section will be drawn from McKinney's book, *Python for Data Analysis*.)

The name 'pandas' actually has nothing to do with Chinese bears but rather comes from the term *panel data*, a form of multi-dimensional data involving measurements over time that comes out the econometrics and statistics community. Ironically, while panel data is a usable data structure in pandas, it is not generally used today and we will not examine it in this course. Instead, we will focus on the two most widely used data structures in pandas: `Series` and `DataFrame`s.

## Reminders about importing and documentation

Just as you imported NumPy undwither the alias ``np``, we will import Pandas under the alias ``pd``:


```python
import pandas as pd
```

As with the NumPy convention, `pd` is an important and widely used convention in the data science world; we will use it here and we advise you to use it in your own coding.

As we progress through Section 5, don't forget that IPython provides tab-completion feature and function documentation with the ``?`` character. If you don't understand anything about a function you see in this section, take a moment and read the documentation; it can help a great deal. As a reminder, to display the built-in pandas documentation, use this code:

```ipython
In [4]: pd?
```

Because it can be useful to lean about `Series` and `DataFrame`s in pandas a extension of `ndarray`s in NumPy, go ahead also import NumPy; you will want it for some of the examples later on:


```python
import numpy as np
```

Now, on to pandas!

## Fundamental panda data structures

Both `Series` and `DataFrame`s are a lot like they `ndarray`s you encountered in the last section. They provide clean, efficent data storage and handling at the scales necessary for data science. What both of them provide that `ndarray`s lack, however, are essential data-science features like flexibility when dealing with missing data and the ability to label data. These capabilities (along with others) help make `Series` and `DataFrame`s essential to the "data munging" that make up so much of data science.

### `Series` objects in pandas

A pandas `Series` is a lot like an `ndarray` in NumPy: a one-dimensional array of indexed data.
You can create a simple Series from an array of data like this:


```python
series_example = pd.Series([-0.5, 0.75, 1.0, -2])
series_example
```

Similar to an `ndarray`, a `Series` upcasts entries to be of the same type of data (that `-2` integer in the original array became a `-2.00` float in the `Series`).

What is different from an `ndarray` is that the ``Series`` automatically wraps both a sequence of values and a sequence of indices. These are two separate objects within the `Seriers` object that can access with the ``values`` and ``index`` attributes.

Try accessing the ``values`` first; they are just a familiar NumPy array:


```python
series_example.values
```

The ``index`` is also an array-like object:


```python
series_example.index
```

Just as with `ndarra`s, you can access specific data elements in a `Series` via the familiar Python square-bracket index notation and slicing:


```python
series_example[1]
```


```python
series_example[1:3]
```

Despite a lot of similarities, pandas `Series` have an important distinction from NumPy `ndarrays`: whereas `ndarrays` have  *implicitly defined* integer indices (as do Python lists), pandas `Series` have *explicitly defined* indices. The best part is that you can set the index:


```python
series_example2 = pd.Series([-0.5, 0.75, 1.0, -2], index=['a', 'b', 'c', 'd'])
series_example2
```

These explicit indices work exactly the way you would expect them to:


```python
series_example2['b']
```

### Exercise:


```python
# Do explicit Series indices work *exactly* the way you might expect?
# Try slicing series_example2 using its explicit index and find out.

```

With explicit indices in the mix, a `Series` is basically a fixed-length, ordered dictionary in that it maps arbitrary typed index values to arbitrary typed data values. But like `ndarray`s these data are all of the same type, which is important. Just as the type-specific compiled code behind `ndarray` makes them more efficient than a Python lists for certain operations, the type information of pandas ``Series`` makes them much more efficient than Python dictionaries for certain operations.

But the connection between `Series` and dictionaries is nevertheless very real: you can construct a ``Series`` object directly from a Python dictionary:


```python
population_dict = {'France': 65429495,
                   'Germany': 82408706,
                   'Russia': 143910127,
                   'Japan': 126922333}
population = pd.Series(population_dict)
population
```

Did you see what happened there? The order of the keys `Russia` and `Japan` in the switched places between the order in which they were entered in `population_dict` and how they ended up in the `population` `Series` object. While Python dictionary keys have no order, `Series` keys are ordered.

So, at one level, you can interact with `Series` as you would with dictionaries:


```python
population['Russia']
```

But you can also do powerful array-like operations with `Series` like slicing:


```python
# Try slicing on the population Series on your own.
# Would slicing be possible if Series keys were not ordered?

```

You can also add elements to a `Series` the way that you would to an `ndarray`. Try it in the code cell below:


```python
# Try running population['Albania'] = 2937590 (or another country of your choice)
# What order do the keys appear in when you run population? Is it what you expected?

```

Anoter useful `Series` feature (and definitely a difference from dictionaries) is that `Series` automatically aligns differently indexed data in arithmetic operations:


```python
pop2 = pd.Series({'Spain': 46432074, 'France': 102321, 'Albania': 50532})
population + pop2
```

Notice that in the case of Germany, Japan, Russia, and Spain (and Albania, depending on what you did in the previous exercise), the addition operation produced `NaN` (not a number) values. pandas does not treat missing values as `0`, but as `NaN` (and it can be helpful to think of arithmetic operations involving `NaN` as essentially `NaN`$ + x=$ `NaN`).

### `DataFrame` object in pandas

The other crucial data structure in pandas to get to know for data science is the `DataFrame`.
Like the ``Series`` object, ``DataFrame``s can be thought of either as generalizations of `ndarray`s (or as specializations of Python dictionaries).

Just as a ``Series`` is like a one-dimensional array with flexible indices, a ``DataFrame`` is like a two-dimensional array with both flexible row indices and flexible column names. Essentially, a `DataFrame` represents a rectangular table of data and contains an ordered collection of labeled columns, each of which can be a different value type (`string`, `int`, `float`, etc.).
The DataFrame has both a row and column index; in this way you can think of it as a dictionary of `Series`, all of which share the same index.

Let's take a look at how this works in practice. We will start by creating a `Series` called `area`:


```python
area_dict = {'Albania': 28748,
             'France': 643801,
             'Germany': 357386,
             'Japan': 377972,
             'Russia': 17125200}
area = pd.Series(area_dict)
area
```

Now you can combine this with the `population` `Series` you created earlier by using a dictionary to construct a single two-dimensional table containing data from both `Series`:


```python
countries = pd.DataFrame({'Population': population, 'Area': area})
countries
```

As with `Series`, note that `DataFrame`s also automatically order indices (in this case, the column indices `Area` and `Population`).

So far we have combined dictionaries together to compose a `DataFrame` (which has given our `DataFrame` a row-centric feel), but you can also create `DataFrame`s in a column-wise fashion. Consider adding a `Capital` column using our reliable old array-analog, a list:


```python
countries['Capital'] = ['Tirana', 'Paris', 'Berlin', 'Tokyo', 'Moscow']
countries
```

As with `Series`, even though initial indices are ordered in `DataFrame`s, subsequent additions to a `DataFrame` stay in the ordered added. However, you can explicitly change the order of `DataFrame` column indices this way:


```python
countries = countries[['Capital', 'Area', 'Population']]
countries
```

Commonly in a data science context, it is necessary to generate new columns of data from existing data sets. Because `DataFrame` columns behave like `Series`, you can do this is by performing operations on them as you would with `Series`:


```python
countries['Population Density'] = countries['Population'] / countries['Area']
countries
```

Note: don't worry if IPython gives you a warning over this. The warning is IPython trying to be a little too helpful. The new column you created is an actual part of the `DataFrame` and not a copy of a slice.

We have stated before that `DataFrame`s are like dictionaries, and it's true. You can retrieve the contents of a column just as you would the value for a specific key in an ordinary dictionary:


```python
countries['Area']
```

What about using the row indices?


```python
# Now try accessing row data with a command like countries['Japan']

```

This returns an error: `DataFrame`s are dictionaries of `Series`, which are the columns. `DataFrame` rows often have heterogeneous data types, so different methods are necessary to access row data. For that, we use the `.loc` method:


```python
countries.loc['Japan']
```

Note that what `.loc` returns is an indexed object in its own right and you can access elements within it using familiar index syntax:


```python
countries.loc['Japan']['Area']
```


```python
# Can you think of a way to return the area of Japan without using .iloc?
# Hint: Try putting the column index first.
# Can you slice along these indices as well?

```

Sometimes it is helpful in data science projects to add a column to a `DataFrame` without assigning values to it:


```python
countries['Debt-to-GDP Ratio'] = np.nan
countries
```

Again, you can disregard the warning (if it triggers) about adding the column this way.

You can also add columns to a `DataFrame` that do not have the same number of rows as the `DataFrame`:


```python
debt = pd.Series([0.19, 2.36], index=['Russia', 'Japan'])
countries['Debt-to-GDP Ratio'] = debt
countries
```

You can use the `del` command to delete a column from a `DataFrame`:


```python
del countries['Capital']
countries
```

In addition to their dictionary-like behavior, `DataFrames` also behave like two-dimensional arrays. For example, it can be useful at times when working with a `DataFrame` to transpose it:


```python
countries.T
```

Again, note that `DataFrame` columns are `Series` and thus the data types must consistent, hence the upcasting to floating-point numbers. **If there had been strings in this `DataFrame`, everything would have been upcast to strings.** Use caution when transposing `DataFrame`s.

#### From a two-dimensional NumPy array

Given a two-dimensional array of data, we can create a ``DataFrame`` with any specified column and index names.
If omitted, an integer index will be used for each:


```python
pd.DataFrame(np.random.rand(3, 2),
             columns=['foo', 'bar'],
             index=['a', 'b', 'c'])
```

## Manipulating data in pandas

A huge part of data science is manipulating data in order to analyze it. (One rule of thumb is that 80% of any data science project will be concerned with cleaning and organizing the data for the project.) So it makes sense to lear the tools that pandas provides for handling data in `Series` and especially `DataFrame`s. Because both of those data structures are ordered, let's first start by taking a closer look at what gives them their structure: the `Index`.

### Index objects in pandas

Both ``Series`` and ``DataFrame``s in pandas have explicit indices that enable you to reference and modify data in them. These indices are actually objects themselves. The ``Index`` object can be thought of as both an immutable array or as fixed-size set. 

It's worth the time to get to know the properties of the `Index` object. Let's return to an example from earlier in the section to examine these properties.


```python
series_example = pd.Series([-0.5, 0.75, 1.0, -2], index=['a', 'b', 'c', 'd'])
ind = series_example.index
ind
```

The ``Index`` works a lot like an array. we have already seen how to use standard Python indexing notation to retrieve values or slices:


```python
ind[1]
```


```python
ind[::2]
```

But ``Index`` objects are immutable; you cannot be modified via the normal means:


```python
ind[1] = 0
```

This immutability is a good thing: it makes it safer to share indices between multiple ``Series`` or ``DataFrame``s without the potential for problems arising from inadvertent index modification.

In addition to being array-like, a Index also behaves like a fixed-size set, including following many of the conventions used by Python's built-in ``set`` data structure, so that unions, intersections, differences, and other combinations can be computed in a familiar way. Let's play around with this to see it in action.


```python
ind_odd = pd.Index([1, 3, 5, 7, 9])
ind_prime = pd.Index([2, 3, 5, 7, 11])
```

In the code cell below, try out the intersection (`ind_odd &amp; ind_prime`), union (`ind_odd | ind_prime`), and the symmetric difference (`ind_odd ^ ind_prime`) of `ind_odd` and `ind_prime`.


```python

```

These operations may also be accessed via object methods, for example ``ind_odd.intersection(ind_prime)``. Below is a table listing some useful `Index` methods and properties.

| **Method**     | **Description**                                                                           |
|:---------------|:------------------------------------------------------------------------------------------|
| [`append`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.append.html)       | Concatenate with additional `Index` objects, producing a new `Index`                      |
| [`diff`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.diff.html)         | Compute set difference as an Index                                                        |
| [`drop`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html)         | Compute new `Index` by deleting passed values                                             |
| [`insert`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.insert.html)       | Compute new `Index` by inserting element at index `i`                                     |
| [`is_monotonic`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.is_monotonic.html) | Returns `True` if each element is greater than or equal to the previous element           |
| [`is_unique`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.is_unique.html)    | Returns `True` if the Index has no duplicate values                                       |
| [`isin`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.isin.html)         | Compute boolean array indicating whether each value is contained in the passed collection |
| [`unique`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.unique.html)       | Compute the array of unique values in order of appearance                                         |

### Data Selection in Series

As a refresher, a ``Series`` object acts in many ways like both a one-dimensional `ndarray` and a standard Python dictionary.

Like a dictionary, the ``Series`` object provides a mapping from a collection of arbitrary keys to a collection of arbitrary values. Back to an old example:


```python
series_example2 = pd.Series([-0.5, 0.75, 1.0, -2], index=['a', 'b', 'c', 'd'])
series_example2
```


```python
series_example2['b']
```

You can also examine the keys/indices and values using dictionary-like Python tools:


```python
'a' in series_example2
```


```python
series_example2.keys()
```


```python
list(series_example2.items())
```

As with dictionaries, you can extend a dictionary by assigning to a new key, you can extend a ``Series`` by assigning to a new index value:


```python
series_example2['e'] = 1.25
series_example2
```

#### Series as one-dimensional array

Because ``Series`` also provide array-style functionality, you can use the NumPy techniques we looked at in Section 3 like slices, masking, and fancy indexing:


```python
# Slicing using the explicit index
series_example2['a':'c']
```


```python
# Slicing using the implicit integer index
series_example2[0:2]
```


```python
# Masking
series_example2[(series_example2 > -1) & (series_example2 < 0.8)]
```


```python
# Fancy indexing
series_example2[['a', 'e']]
```

One note to avoid confusion. When slicing with an explicit index (i.e., ``series_example2['a':'c']``), the final index is **included** in the slice; when slicing with an implicit index (i.e., ``series_example2[0:2]``), the final index is **excluded** from the slice.

#### Indexers: `loc` and `iloc`

A great thing about pandas is that you can use a lot different things for your explicit indices. A potentially confusing thing about pandas is that you can use a lot different things for your explicit indices, including integers. To avoid confusion between integer indices that you might supply and those implicit integer indices that pandas generates, pandas provides special *indexer* attributes that explicitly expose certain indexing schemes.

(A technical note: These are not functional methods; they are attributes that expose a particular slicing interface to the data in the ``Series``.)

The ``loc`` attribute allows indexing and slicing that always references the explicit index:


```python
series_example2.loc['a']
```


```python
series_example2.loc['a':'c']
```

The ``iloc`` attribute enables indexing and slicing using the implicit, Python-style index:


```python
series_example2.iloc[0]
```


```python
series_example2.iloc[0:2]
```

A guiding principle of the Python language is the idea that "explicit is better than implicit." Professional code will generally use explicit indexing with ``loc`` and ``iloc`` and you should as well in order to make your code clean and readable.

### Data selection in DataFrames

``DataFrame``s also exhibit dual behavior, acting both like a two-dimensional `ndarray` and like a dictionary of ``Series``  sharing the same index.

#### DataFrame as dictionary of Series

Let's return to our earlier example of countries' areas and populations in order to examine `DataFrame`s as a dictionary of `Series`.


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

You can access the individual ``Series`` that make up the columns of a ``DataFrame`` via dictionary-style indexing of the column name:


```python
countries['Area']
```

An you can use dictionary-style syntax can also be used to modify `DataFrame`s, such as by adding a new column:


```python
countries['Population Density'] = countries['Population'] / countries['Area']
countries
```

#### DataFrame as two-dimensional array

You can also think of ``DataFrame``s as two-dimensional arrays. You can examine the raw data in the `DataFrame`/data array using the ``values`` attribute:


```python
countries.values
```

Viewed thsi way it makes sense that we can transpose the rows and columns of a `DataFrame` the same way we would an array:


```python
countries.T
```

`DataFrame`s also uses the ``loc`` and ``iloc`` indexers. With ``iloc``, you can index the underlying array as if it were an `ndarray` but with the ``DataFrame`` index and column labels maintained in the result:


```python
countries.iloc[:3, :2]
```

``loc`` also permits array-like slicing but using the explicit index and column names:


```python
countries.loc[:'Germany', :'Population']
```

You can also use array-like techniques such as masking and fancing indexing with `loc`.


```python
# Can you think of how to combine masking and fancy indexing in one line?
# Your masking could be somthing like countries['Population Density'] > 200
# Your fancy indexing could be something like ['Population', 'Population Density']
# Be sure to put the the masking and fancy indexing inside the square brackets: countries.loc[]

```

#### Indexing conventions

In practice in the world of data science (and pandas more generally), *indexing* refers to columns while *slicing* refers to rows:


```python
countries['France':'Japan']
```

Such slices can also refer to rows by number rather than by index:


```python
countries[1:3]
```

Similarly, direct masking operations are also interpreted row-wise rather than column-wise:


```python
countries[countries['Population Density'] > 200]
```

These two conventions are syntactically similar to those on a NumPy array, and while these may not precisely fit the mold of the Pandas conventions, they are nevertheless quite useful in practice.

# Operating on Data in Pandas

As you begin to work in data science, operating on data is imperative. It is the very heart of data science. Another aspect of pandas that makes it a compelling tool for many data scientists is pandas' capability to perform efficient element-wise operations on data. pandas builds on ufuncs from NumPy to supply theses capabilities and then extends them to provide additional power for data manipulation:
 - For unary operations (such as negation and trigonometric functions), ufuncs in pandas **preserve index and column labels** in the output.
 - For binary operations (such as addition and multiplication), pandas automatically **aligns indices** when passing objects to  ufuncs.

These critical features of ufuncs in pandas mean that data retains its context when operated on and, more importantly still, drastically helps reduce errors when you combine data from multiple sources.

## Index Preservation

pandas is explicitly designed to work with NumPy. As a results, all NumPy ufuncs will work on Pandas ``Series`` and ``DataFrame`` objects.

We can see this more clearly if we create a simple ``Series`` and ``DataFrame`` of random numbers on which to operate. 


```python
rng = np.random.RandomState(42)
ser_example = pd.Series(rng.randint(0, 10, 4))
ser_example
```

Did you notice the NumPy function we used with the variable `rng`? By specifying a seed for the random-number generator, you get the same result each time. This can be useful trick when you need to produce psuedo-random output that also needs to be replicatable by others. (Go ahead and re-run the code cell above a couple of times to convince yourself that it produces the same output each time.)


```python
df_example = pd.DataFrame(rng.randint(0, 10, (3, 4)),
                  columns=['A', 'B', 'C', 'D'])
df_example
```

Let's apply a ufunc to our example `Series`:


```python
np.exp(ser_example)
```

The same thing happens with a slightly more complex operation on our example `DataFrame`:


```python
np.cos(df_example * np.pi / 4)
```

Note that you can use all of the ufuncs we discussed in Section 3 the same way.

## Index alignment

As mentioned above, when you perform a binary operation on two ``Series`` or ``DataFrame`` objects, pandas will align indices in the process of performing the operation. This is essential when working with incomplete data (and data is usually incomplete), but it is helpful to see this in action to better understand it.

### Index alignment with Series

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

```

Your resulting array contains the **union** of indices of the two input arrays: seven countries in total. All of the countries in the array without an entry (because they lacked either area data or population data) are marked with the now familiar ``NaN``, or "Not a Number," designation.

Index matching works the same way built-in Python arithmetic expressions and missing values are filled in with `NaN`s. You can see this clearly by adding two `Series` that are slightly misaligned in their indices:


```python
series1 = pd.Series([2, 4, 6], index=[0, 1, 2])
series2 = pd.Series([3, 5, 7], index=[1, 2, 3])
series1 + series2
```

`NaN` values are not always convenient to work with; `NaN` combined with any other values results in `NaN`, which can be a pain, particulalry if you are combining multiple data sources with missing values. To help with this, pandas allows you to specify a default value to use for missing values in the operation. For example, calling `series1.add(series2)` is equivalent to calling `series1 + series2`, but you can supply the fill value:


```python
series1.add(series2, fill_value=0)
```

Much better!

### Index alignment with DataFrames

The same kind of alignment takes place in both dimension (columns and indices) when you perform operations on ``DataFrame``s.


```python
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
```

Even though we passed the columns in a different order in `df2` than in `df1`, the indices were aligned correctly sorted in the resulting union of columns.

You can also use fill values for missing values with `Data Frame`s. In this example, let's fill the missing values with the mean of all values in `df1` (computed by first stacking the rows of `df1`):


```python
fill = df1.stack().mean()
df1.add(df2, fill_value=fill)
```

This table lists Python operators and their equivalent pandas object methods:

| Python Operator | Pandas Method(s)                      |
|-----------------|---------------------------------------|
| ``+``           | ``add()``                             |
| ``-``           | ``sub()``, ``subtract()``             |
| ``*``           | ``mul()``, ``multiply()``             |
| ``/``           | ``truediv()``, ``div()``, ``divide()``|
| ``//``          | ``floordiv()``                        |
| ``%``           | ``mod()``                             |
| ``**``          | ``pow()``                             |


## Operations between DataFrames and Series

Index and column alignment gets maintained in operations between a `DataFrame` and a `Series` as well. To see this, consider a common operation in data science, wherein we find the difference of a `DataFrame` and one of its rows. Because pandas inherits ufuncs from NumPy, pandas will compute the difference row-wise by default:


```python
df3 = pd.DataFrame(rng.randint(10, size=(3, 4)), columns=list('WXYZ'))
df3
```


```python
df3 - df3.iloc[0]
```

But what if you need to operate column-wise? You can do this by using object methodsand specifying the ``axis`` keyword.


```python
df3.subtract(df3['X'], axis=0)
```

And when you do operations between `DataFrame`s and `Series` operations, you still get automatic index alignment:


```python
halfrow = df3.iloc[0, ::2]
halfrow
```

Note that the output from that operation was transposed. That was so that we can subtract it from the `DataFrame`:


```python
df3 - halfrow
```

Remember, pandas preserves and aligns indices and columns so preserve data context. This will be of huge help to you in our next section when we look at data cleaning and preparation.
