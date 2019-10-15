
# Introduction to NumPy

**Library Alias**


```python
import numpy as np
```

## Built-In Help


### Exercise


```python
# Place your cursor after the period and press <TAB>:
np.
```



### Exercise


```python
# Replace 'add' below with a few different NumPy function names and look over the documentation:
np.add?
```

## NumPy arrays: a specialized data structure for analysis

&gt; **Learning goal:** By the end of this subsection, you should have a basic understanding of what NumPy arrays are and how they differ from the other Python data structures you have studied thus far.

### Lists in Python



```python
myList = list(range(10))
myList
```

**List Comprehension with Types**


```python
[type(item) for item in myList]
```

**Share**


```python
myList2 = [True, "2", 3.0, 4]
[type(item) for item in myList2]
```

### Fixed-type arrays in Python

#### Creating NumPy arrays method 1: using Python lists


```python
# Create an integer array:
np.array([1, 4, 2, 5, 3])
```

**Think, Pair, Share**


```python
np.array([3.14, 4, 2, 3])
```

### Exercise


```python
# What happens if you construct an array using a list that contains a combination of integers, floats, and strings?

```

**Explicit Typing**


```python
np.array([1, 2, 3, 4], dtype='float32')
```

### Exercise


```python
# Try this using a different dtype.
# Remember that you can always refer to the documentation with the command np.array.

```

**Multi-Dimensional Array**

**Think, Pair, Share**


```python
# nested lists result in multi-dimensional arrays
np.array([range(i, i + 3) for i in [2, 4, 6]])
```

#### Creating NumPy arrays method 2: building from scratch


```python
np.zeros(10, dtype=int)
```


```python
np.ones((3, 5), dtype=float)
```


```python
np.full((3, 5), 3.14)
```


```python
np.arange(0, 20, 2)
```


```python
np.linspace(0, 1, 5)
```


```python
np.random.random((3, 3))
```


```python
np.random.normal(0, 1, (3, 3))
```


```python
np.random.randint(0, 10, (3, 3))
```


```python
np.eye(3)
```


```python
np.empty(3)
```

&gt; **Takeaway:** NumPy arrays are a data structure similar to Python lists that provide high performance when storing and working on large amounts of homogeneous data—precisely the kind of data that you will encounter frequently in doing data science. NumPy arrays support many data types beyond those discussed in this course. With all of that said, however, don’t worry about memorizing all of the NumPy dtypes. **It’s often just necessary to care about the general kind of data you’re dealing with: floating point, integer, Boolean, string, or general Python object.**

## Working with NumPy arrays: the basics

&gt; **Learning goal:** By the end of this subsection, you should be comfortable working with NumPy arrays in basic ways.

**Similar to Lists:**
- **Arrays attributes**: Assessing the size, shape, and data types of arrays
- **Indexing arrays**: Getting and setting the value of individual array elements
- **Slicing arrays**: Getting and setting smaller subarrays within a larger array
- **Reshaping arrays**: Changing the shape of a given array
- **Joining and splitting arrays**: Combining multiple arrays into one and splitting one array into multiple arrays

### Array attributes


```python
import numpy as np
np.random.seed(0)  # seed for reproducibility

a1 = np.random.randint(10, size=6)  # One-dimensional array
a2 = np.random.randint(10, size=(3, 4))  # Two-dimensional array
a3 = np.random.randint(10, size=(3, 4, 5))  # Three-dimensional array
```

**Array Types**


```python
print("dtype:", a3.dtype)
```



### Exercise:


```python
# Change the values in this code snippet to look at the attributes for a1, a2, and a3:
print("a3 ndim: ", a3.ndim)
print("a3 shape:", a3.shape)
print("a3 size: ", a3.size)
```

### Exercise:


```python
# Explore the dtype for the other arrays.
# What dtypes do you predict them to have?
print("dtype:", a3.dtype)
```

### Indexing arrays

**Quick Review**


```python
a1
```


```python
a1[0]
```


```python
a1[4]
```


```python
a1[-1]
```


```python
a1[-2]
```

**Multi-Dimensional Arrays**


```python
a2
```


```python
a2[0, 0]
```


```python
a2[2, 0]
```


```python
a2[2, -1]
```


```python
a2[0, 0] = 12
a2
```


```python
a1[0] = 3.14159
a1
```

### Exercise:


```python
# What happens if you try to insert a string into a1?
# Hint: try both a string like '3' and one like 'three'

```

### Slicing arrays

#### One-dimensional slices


```python
a = np.arange(10)
a
```


```python
a[:5]
```


```python
a[5:]
```


```python
a[4:7]
```

**Slicing With Index**


```python
a[::2]
```


```python
a[1::2]
```


```python
a[::-1]
```


```python
a[5::-2]
```

#### Multidimensional slices


```python
a2
```


```python
a2[:2, :3]
```


```python
a2[:3, ::2]
```


```python
a2[::-1, ::-1]
```

#### Accessing array rows and columns


```python
print(a2[:, 0])
```


```python
print(a2[0, :])
```


```python
print(a2[0])
```

#### Slices are no-copy views


```python
print(a2)
```


```python
a2_sub = a2[:2, :2]
print(a2_sub)
```


```python
a2_sub[0, 0] = 99
print(a2_sub)
```


```python
print(a2)
```

#### Copying arrays



```python
a2_sub_copy = a2[:2, :2].copy()
print(a2_sub_copy)
```


```python
a2_sub_copy[0, 0] = 42
print(a2_sub_copy)
```


```python
print(a2)
```

### Joining and splitting arrays

#### Joining arrays


```python
a = np.array([1, 2, 3])
b = np.array([3, 2, 1])
np.concatenate([a, b])
```


```python
c = [99, 99, 99]
print(np.concatenate([a, b, c]))
```


```python
grid = np.array([[1, 2, 3],
                 [4, 5, 6]])
```


```python
np.concatenate([grid, grid])
```

#### Splitting arrays
**Think, Pair, Share**


```python
a = [1, 2, 3, 99, 99, 3, 2, 1]
a1, a2, a3 = np.split(a, [3, 5])
print(a1, a2, a3)
```

&gt; **Takeaway:** Manipulating datasets is a fundamental part of preparing data for analysis. The skills you learned and practiced here will form building blocks for the most sophisticated data-manipulation you will learn in later sections in this course.

## Sorting arrays


```python
a = np.array([2, 1, 4, 3, 5])
np.sort(a)
```


```python
print(a)
```


```python
a.sort()
print(a)
```

### Sorting along rows or columns


```python
rand = np.random.RandomState(42)
table = rand.randint(0, 10, (4, 6))
print(table)
```


```python
np.sort(table, axis=0)
```


```python
np.sort(table, axis=1)
```

### NumPy Functions vs Python Built-In Functions

| Operator	    | Equivalent ufunc    | Description                           |
|:--------------|:--------------------|:--------------------------------------|
|``+``          |``np.add``           |Addition (e.g., ``1 + 1 = 2``)         |
|``-``          |``np.subtract``      |Subtraction (e.g., ``3 - 2 = 1``)      |
|``-``          |``np.negative``      |Unary negation (e.g., ``-2``)          |
|``*``          |``np.multiply``      |Multiplication (e.g., ``2 * 3 = 6``)   |
|``/``          |``np.divide``        |Division (e.g., ``3 / 2 = 1.5``)       |
|``//``         |``np.floor_divide``  |Floor division (e.g., ``3 // 2 = 1``)  |
|``**``         |``np.power``         |Exponentiation (e.g., ``2 ** 3 = 8``)  |
|``%``          |``np.mod``           |Modulus/remainder (e.g., ``9 % 4 = 1``)|

#### Exponents and logarithms


```python
a = [1, 2, 3]
print("a     =", a)
print("e^a   =", np.exp(a))
print("2^a   =", np.exp2(a))
print("3^a   =", np.power(3, a))
```


```python
a = [1, 2, 4, 10]
print("a        =", a)
print("ln(a)    =", np.log(a))
print("log2(a)  =", np.log2(a))
print("log10(a) =", np.log10(a))
```


```python
a = [0, 0.001, 0.01, 0.1]
print("exp(a) - 1 =", np.expm1(a))
print("log(1 + a) =", np.log1p(a))
```

#### Specialized Functions


```python
from scipy import special
```


```python
# Gamma functions (generalized factorials) and related functions
a = [1, 5, 10]
print("gamma(a)     =", special.gamma(a))
print("ln|gamma(a)| =", special.gammaln(a))
print("beta(a, 2)   =", special.beta(a, 2))
```

&gt; **Takeaway:** Universal functions in NumPy provide you with computational functions that are faster than regular Python functions, particularly when working on large datasets that are common in data science. This speed is important because it can make you more efficient as a data scientist and it makes a broader range of inquiries into your data tractable in terms of time and computational resources.

## Aggregations

&gt; **Learning goal:** By the end of this subsection, you should be comfortable aggregating data in NumPy.

### Summing the values of an array


```python
myList = np.random.random(100)
np.sum(myList)
```

**NumPy vs Python Functions**


```python
large_array = np.random.rand(1000000)
%timeit sum(large_array)
%timeit np.sum(large_array)
```

### Minimum and maximum


```python
np.min(large_array), np.max(large_array)
```


```python
print(large_array.min(), large_array.max(), large_array.sum())
```

## Computation on arrays with broadcasting

&gt; **Learning goal:** By the end of this subsection, you should have a basic understanding of how broadcasting works in NumPy (and why NumPy uses it).


```python
first_array = np.array([3, 6, 8, 1])
second_array = np.array([4, 5, 7, 2])
first_array + second_array
```


```python
first_array + 5
```


```python
one_dim_array = np.ones((1))
one_dim_array
```


```python
two_dim_array = np.ones((2, 2))
two_dim_array
```


```python
one_dim_array + two_dim_array
```

**Think, Pair, Share**


```python
horizontal_array = np.arange(3)
vertical_array = np.arange(3)[:, np.newaxis]

print(horizontal_array)
print(vertical_array)
```


```python
horizontal_array + vertical_array
```

## Comparisons, masks, and Boolean logic in NumPy

&gt; **Learning goal:** By the end of this subsection, you should be comfortable with and understand how to use Boolean masking in NumPy in order to answer basic questions about your data.

### Example: Counting Rainy Days

Let's see masking in practice by examining the monthly rainfall statistics for Seattle. The data is in a CSV file from data.gov. To load the data, we will use pandas, which we will formally introduce in Section 4.


```python
import numpy as np
import pandas as pd

# Use pandas to extract rainfall as a NumPy array
rainfall_2003 = pd.read_csv('Data/Observed_Monthly_Rain_Gauge_Accumulations_-_Oct_2002_to_May_2017.csv')['RG01'][ 2:14].values
rainfall_2003
```


```python
%matplotlib inline
import matplotlib.pyplot as plt
```


```python
plt.bar(np.arange(1, len(rainfall_2003) + 1), rainfall_2003)
```

### Boolean operators


```python
np.sum((rainfall_2003 > 0.5) & (rainfall_2003 < 1))
```


```python
rainfall_2003 > (0.5 & rainfall_2003) < 1
```


```python
np.sum(~((rainfall_2003 <= 0.5) | (rainfall_2003 >= 1)))
```


```python
print("Number of months without rain:", np.sum(rainfall_2003 == 0))
print("Number of months with rain:   ", np.sum(rainfall_2003 != 0))
print("Months with more than 1 inch: ", np.sum(rainfall_2003 > 1))
print("Rainy months with < 1 inch:   ", np.sum((rainfall_2003 > 0) &
                                              (rainfall_2003 < 1)))
```

## Boolean arrays as masks


```python
rand = np.random.RandomState(0)
two_dim_array = rand.randint(10, size=(3, 4))
two_dim_array
```


```python
two_dim_array < 5
```

**Masking**


```python
two_dim_array[two_dim_array < 5]
```


```python
# Construct a mask of all rainy months
rainy = (rainfall_2003 > 0)

# Construct a mask of all summer months (June through September)
months = np.arange(1, 13)
summer = (months > 5) & (months < 10)

print("Median precip in rainy months in 2003 (inches):   ", 
      np.median(rainfall_2003[rainy]))
print("Median precip in summer months in 2003 (inches):  ", 
      np.median(rainfall_2003[summer]))
print("Maximum precip in summer months in 2003 (inches): ", 
      np.max(rainfall_2003[summer]))
print("Median precip in non-summer rainy months (inches):", 
      np.median(rainfall_2003[rainy & ~summer]))
```

&gt; **Takeaway:** By combining Boolean operations, masking operations, and aggregates, you can quickly answer questions similar to those we posed about the Seattle rainfall data about any dataset. Operations like these will form the basis for the data exploration and preparation for analysis that will by our primary concerns in Sections 4 and 5.
