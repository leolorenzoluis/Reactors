
# Introduction to Python

## Comments


```python
# this is the first comment
spam = 1  # and this is the second comment
          # ... and now a third!
text = "# This is not a comment because it's inside quotes."
print(text)
```

## Python basics

### Arithmetic and numeric types

&gt; **Learning goal:** By the end of this subsection, you should be comfortable with using numeric types in Python arithmetic.

#### Python numeric operators


```python
2 + 3
```

**Share**: What is the answer? Why?


```python
30 - 4 * 5
```

**Share**: What is the answer? Why?


```python
7 / 5
```


```python
3 * 3.5
```


```python
7.0 / 5
```

**Floor Division**


```python
7 // 5
```

**Remainder (modulo)**


```python
7 % 5
```

**Exponents**


```python
5 ** 2
```


```python
2 ** 5
```

**Share**: What is the answer? Why?


```python
-5 ** 2
```


```python
(-5) ** 2
```


```python
(30 - 4) * 5
```

### Variables

**Share**: What is the answer? Why?


```python
length = 15
width = 3 * 5
length * width
```

**Variables don't need types**


```python
length = 15
length
```


```python
length = 15.0
length
```


```python
length = 'fifteen'
length
```

**Share**: What will happen? Why?


```python
n
```

**Previous Output**


```python
tax = 11.3 / 100
price = 19.95
price * tax
```


```python
price + _
```


```python
round(_, 2)
```

**Multiple Variable Assignment**


```python
a, b, c, = 3.2, 1, 6
a, b, c
```

### Expressions

**Share**: What is the answer? Why?


```python
2 < 5
```

*(Run after learners have shared the above)*
**Python Comparison Operators**:
![all of the comparison operators](https://notebooks.azure.com/sguthals/projects/data-science-1-instructor/raw/Images%2FScreen%20Shot%202019-09-10%20at%207.15.49%20AM.png)

**Complex Expressions**


```python
a, b, c = 1, 2, 3
a < b < c
```

**Built-In Functions**


```python
min(3, 2.4, 5)
```


```python
max(3, 2.4, 5)
```

**Compound Expressions**


```python
1 < 2 and 2 < 3
```



### Exercise:

**Think, Pair, Share**
1. Quietly think about what would happen if you flipped one of the `&lt;` to a `&gt;`. 
2. Share with the person next to you what you think will happen. 
3. Try it out in the code cell below. 
4. Share anything you thought was surprising.


```python
# Now flip around one of the simple expressions and see if the output matches your expectations:

```

**Or and Not**  
**Share**: What is the answer? Why?


```python
1 < 2 or 1 > 2
```


```python
not (2 < 3)
```

### Exercise:
**Think, Pair, Share**
1. Quietly think about what would the results would be. *Tip: Use paper!*
2. Share with the person next to you what you think will happen. 
3. Try it out in the code cell below. 
4. Share anything you thought was surprising.
5. Instructor Demo


```python
# Play around with compound expressions.
# Set i to different values to see what results this complex compound expression returns:
i = 7
(i == 2) or not (i % 2 != 0 and 1 < i < 5)
```

&gt; **Takeaway:** Arithmetic operations on numeric data form the foundation of data science work in Python. Even sophisticated numeric operations are predicated on these basics, so mastering them is essential to doing data science.

## Strings

&gt; **Learning goal:** By the end of this subsection, you should be comfortable working with strings at a basic level in Python.


```python
'spam eggs'  # Single quotes.
```


```python
'doesn\'t'  # Use \' to escape the single quote...
```


```python
"doesn't"  # ...or use double quotes instead.
```


```python
'"Isn\'t," she said.'
```


```python
print('"Isn\'t," she said.')
```

**Pause**
Notice the difference between the previous two code cells when they are run. 


```python
print('C:\some\name')  # Here \n means newline!
```


```python
print(r'C:\some\name')  # Note the r before the quote.
```

### String literals

**Think, Pair, Share**


```python
3 * 'un' + 'ium'
```

### Concatenating strings


```python
'Py' 'thon'
```


```python
prefix = 'Py'
prefix + 'thon'
```

### String indexes

**Think, Pair, Share**


```python
word = 'Python'
word[0]
```

**Share**


```python
word[5]
```

**Share**


```python
word[-1]
```


```python
word[-2]
```


```python
word[-6]
```

### Slicing strings

**Think, Pair, Share**


```python
word[0:2]
```

**Share**


```python
word[2:5]
```


```python
word[:2]
```


```python
word[4:]
```


```python
word[-2:]
```

**Share**


```python
word[:2] + word[2:]
```


```python
word[:4] + word[4:]
```

**TIP**
 +---+---+---+---+---+---+
 | P | y | t | h | o | n |
 +---+---+---+---+---+---+
 0   1   2   3   4   5   6
-6  -5  -4  -3  -2  -1
**Share**


```python
word[42]  # The word only has 6 characters.
```

**Share**


```python
word[4:42]
```


```python
word[42:]
```

**Strings are Immutable**


```python
word[0] = 'J'
```


```python
word[2:] = 'py'
```


```python
'J' + word[1:]
```


```python
word[:2] + 'Py'
```

**Built-In Function: len**


```python
s = 'supercalifragilisticexpialidocious'
len(s)
```

**Built-In Function: str**


```python
str(2)
```


```python
str(2.5)
```

## Other data types

&gt; **Learning goal:** By the end of this subsection, you should have a basic understanding of the remaining fundamental data types in Python and an idea of how and when to use them.

### Lists


```python
squares = [1, 4, 9, 16, 25]
squares
```

**Indexing and Slicing is the Same as Strings**


```python
squares[0] 
```


```python
squares[-1]
```


```python
squares[-3:]
```


```python
squares[:]
```

**Think, Pair, Share**


```python
squares + [36, 49, 64, 81, 100]
```

**Lists are Mutable**


```python
cubes = [1, 8, 27, 65, 125]
4 ** 3 
```

**Think, Pair, Share**


```python
# Replace the wrong value.
cubes
```

**Replace Many Values**


```python
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
letters
```

**Share**


```python
letters[2:5] = ['C', 'D', 'E']
letters
```

**Share**


```python
letters[2:5] = []
letters
```


```python
letters[:] = []
letters
```

**Built-In Functions: len**


```python
letters = ['a', 'b', 'c', 'd']
len(letters)
```

**Nesting**

**Think, Pair, Share**


```python
a = ['a', 'b', 'c']
n = [1, 2, 3]
x = [a, n]
x
```

**Share**


```python
x[0]
```

**Share**


```python
x[0][0]
```

### Exercise:


```python
# Nested lists come up a lot in programming, so it pays to practice.
# Which indices would you include after x to get ‘c’?
# How about to get 3?

```

### List object methods

**Share**


```python
beatles = ['John', 'Paul']
beatles.append('George')
beatles
```

**Share**


```python
beatles2 = ['John', 'Paul', 'George']
beatles2.append(['Stuart', 'Pete'])
beatles2
```

**Share**


```python
beatles.extend(['Stuart', 'Pete'])
beatles
```

**Share**


```python
beatles.index('George')
```


```python
beatles.count('John')
```


```python
beatles.remove('Stuart')
beatles
```


```python
beatles.pop()
```


```python
beatles.insert(1, 'Ringo')
beatles
```


```python
beatles.reverse()
beatles
```


```python
beatles.sort()
beatles
```

### Exercise:


```python
# What happens if you run beatles.extend(beatles)?
# How about beatles.append(beatles)?

```

### Tuples


```python
t = (1, 2, 3)
t
```

**Tuples are Immutable**


```python
t[1] = 2.0
```


```python
t[1]
```


```python
t[:2]
```

**Lists &lt;-&gt; Tuples**


```python
l = ['baked', 'beans', 'spam']
l = tuple(l)
l
```


```python
l = list(l)
l
```

### Membership testing

**Share**


```python
tup = ('a', 'b', 'c')
'b' in tup
```


```python
lis = ['a', 'b', 'c']
'a' not in lis
```

### Exercise:


```python
# What happens if you run lis in lis?
# Is that the behavior you expected?
# If not, think back to the nested lists we’ve already encountered.

```

### Dictionaries


```python
capitals = {'France': ('Paris', 2140526)}
```


```python
capitals['Nigeria'] = ('Lagos', 6048430)
capitals
```

### Exercise:


```python
# Now try adding another country (or something else) to the capitals dictionary
```

**Interacting with Dictionaries**


```python
capitals['France']
```


```python
capitals['Nigeria'] = ('Abuja', 1235880)
capitals
```


```python
len(capitals)
```


```python
capitals.popitem()
```


```python
capitals
```

&gt; **Takeaway:** Regardless of how complex and voluminous the data you will work with, these basic data structures will repeatedly be your means for handling and manipulating it. Comfort with these basic data structures is essential to being able to understand and use Python code written by others.

### List comprehensions

&gt; **Learning goal:** By the end of this subsection, you should understand how to economically and computationally create lists.


```python
for x in range(1,11):
    print(x)
```


```python
numbers = [x for x in range(1,11)] # Remember to create a range 1 more than the number you actually want.
numbers

numbers = [x for x in range(1,11)]
numbers = [x for x in [1,2,3,4,5,6,7,8,9,10]]
numbers = [1,2,3,4,5,6,7,8,9,10]
```


```python
for x in range(1,11):
    print(x*x)
```


```python
squares = [x*x for x in range(1,11)]
squares

squares = [x*x for x in range(1,11)]
squares = [x*x for x in [1,2,3,4,5,6,7,8,9,10]]
squares = [1*1,2*2,3*3,4*4,5,6,7,8,9,10]
squares = [1,2,9...]
```

**Demo**


```python
odd_squares = [x*x for x in range(1,11) if x % 2 != 0]
odd_squares
```

### Exercise:


```python
# Now use a list comprehension to generate a list of odd cubes
# from 1 to 2,197

```

&gt; **Takeaway:** List comprehensions are a popular tool in Python because they enable the rapid, programmatic generation of lists. The economy and ease of use therefore make them an essential tool for you (in addition to a necessary topic to understand as you try to understand Python code written by others).

### Importing modules

&gt; **Learning goal:** By the end of this subsection, you should be comfortable importing modules in Python.


```python
factorial(5)
```


```python
import math
math.factorial(5)
```


```python
from math import factorial
factorial(5)
```


&gt; **Takeaway:** There are several Python modules that you will regularly use in conducting data science in Python, so understanding how to import them will be essential (especially in this training).
