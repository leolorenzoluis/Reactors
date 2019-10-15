
# Introduction to Python

Since its first release in 1991, Python has risen to become not just a popular general-purpose programming language, but a preeminent computer language for data science. This notebook is designed to accompany the Microsoft Reactors introduction to data science workshop, and it will use Python as the primary means of illustrating data-science tools and resources.

Several examples in this notebook draw from the [python.org introductory tutorial](https://docs.python.org/3.5/tutorial/introduction.html) and examples given in the [Python 3 documentation](https://docs.python.org/3/) (with edits and amendments). This introduction to Python is written for Python 3.6.7 but is generally applicable to other Python 3.x versions. 

Original material from python.org is Copyright (c) 2001-2019 Python Software Foundation.

This course makes extensive use of Jupyter Notebooks hosted on Microsoft Azure. Azure-hosted Jupyter Notebooks provide an easy way for you to experiment with programming concepts in an interactive fashion that requires no installation of software by students on local computers.

Jupyter Notebooks are divided into cells. Each cell contains either text written in the Markdown markup language or a space in which to write and execute computer code. Because all the code resides inside code cells, you can run each code cell inline rather than using a separate Python interactive window.

&gt; **Note**: This notebook is designed to have you run code cells one by one, and several code cells contain deliberate errors for demonstration purposes. As a result, if you use the **Cell** &gt; **Run All** command, some code cells past the error won't be run. To resume running the code in each case, use **Cell** &gt; **Run All Below** from the cell after the error.

## Comments

Many of the examples in this notebook include comments. Comments in Python start with the hash character (`#`) and extend to the end of the physical line. A comment may appear at the start of a line or following white space or code, but not within a string literal. A hash character within a string literal is just a hash character. Because comments are there to clarify code and are not interpreted by Python, they may be omitted when typing in examples. For example:


```python
# this is the first comment
spam = 1  # and this is the second comment
          # ... and now a third!
text = "# This is not a comment because it's inside quotes."
print(text)
```

## Python basics

This section will provide you with a foundational understanding of Python syntax and data types.

### Arithmetic and numeric types

&gt; **Learning goal:** By the end of this subsection, you should be comfortable with using numeric types in Python arithmetic.

Python is an interpreted language, which means that you can interactively use the interpreter to get immediate results. You can see this by using the Python interpreter as a simple calculator: type an expression, and you can see the output immediately.

How can you see the results? The Python interpreter runs inside this notebook. To run the code inside a cell, either click the **Run Cell** button at the top of the window or press **Ctrl**+**Enter**. Try running the contents of the cell below. (Don't worry, we'll cover what the syntax of the Python code means later on in this section.)


```python
print("Hello, world.")
```

#### Python numeric operators

Expression syntax is straightforward: the operators `+`, `-`, `*`, and `/` work just like in most other programming languages (such as Java or C). For example:


```python
2 + 3
```

The order of operations also works as in other programming languages (and in math class):


```python
30 - 4 * 5
```

Note what happens when you use division:


```python
7 / 5
```

Division (`/`) always returns a floating-point number, which brings up a good point. Python (like other programming languages) has different numeric types. Integer numbers (such as `1`, `3`, and `20`) have type [`int`](https://docs.python.org/3.6/library/functions.html#int). Numbers with a fractional component (such as `3.0` or `1.6`) have type [`float`](https://docs.python.org/3.5/library/functions.html#float).

You can mix numeric types in calculations:


```python
3 * 3.5
```


```python
7.0 / 5
```

You can perform a type of division that returns an integer: [floor division](https://docs.python.org/3.6/glossary.html#term-floor-division). Floor division uses the `//` operator, discards any remainders, and just returns an `int`.


```python
7 // 5
```

To calculate the remainder, you can use the modulo operator, `%`:


```python
7 % 5
```

For exponents, use the `**` operator. For example, you can write $5^2$ as:


```python
5 ** 2
```

Conversely, $2^5$ would be:


```python
2 ** 5
```

Note that `**` has higher precedence in the order of operations than the negative sign, `-`. This means that $-5^2$ is actually the same thing as $-\left(5^2\right)$:


```python
-5 ** 2
```

In order to assert the order of precedence that you want, use parentheses, `()`:


```python
(-5) ** 2
```

Parentheses can supersede the order of operations in any calculation you need to run:


```python
(30 - 4) * 5
```

### Variables

As in other programming languages, it is often essential to save values for later using variables in Python. Python assigns values to variables using the equals sign (`=`):


```python
length = 15
width = 3 * 5
length * width
```

If you come from a programming background in another programming language (such as Java), you might have noticed that we never specified the variable type when we declared our variables `length` and `width`. Python does not require this, and you can change variable types as you wish:


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

Note that, for all the flexibility of variables in Python, you do have to define them. If you try to use an undefined variable, it will produce an error:


```python
n
```

In Python's interactive mode and in Jupyter Notebooks, you can use the built-in variable `_`, which automatically takes the value of the last printed expression. For example:


```python
tax = 11.3 / 100
price = 19.95
price * tax
```


```python
price + _
```

Note that you should always treat the `_` variable as read-only. Explicitly assigning a value to it will create an independent local variable with the same name and will mask the built-in variable (and its behavior).

Our previous output was kind of a mess, however; we generally use only two or fewer decimal points when working with prices. In order to clean this up, we can use a built-in function, `round()`.


```python
round(_, 2)
```

We will cover some of the other functions built into Python later in this section and cover user-defined functions in Section 2.

You do not have to define variables one at a time. You can define multiple variables on a single line, like so:


```python
a, b, c, = 3.2, 1, 6
a, b, c
```

You can also augment variable assignments. This will be particularly useful when we tackle loops in the next section.


```python
x = 5
x = x + 1  # Un-pythonic variable augmentation
x += 1  # Pythonic variable augmentation
x
```

Note that augmented assignment doesn’t have to be by 1 or even just addition. Beyond +=, augmented assignment statements in Python include -=, \*=, /=, %=, and \**=. Try playing around with different augmentation assignments until this concept makes sense.

Python supports other types of numbers beyond `int` and `float`, such as [`Decimal`](https://docs.python.org/3.6/library/decimal.html#decimal.Decimal) and [`Fraction`](https://docs.python.org/3.6/library/fractions.html#fractions.Fraction). Python also has built-in support for [complex numbers](https://docs.python.org/3.6/library/stdtypes.html#typesnumeric), which are all beyond the scope of this course.

### Expressions

As with other programming languages, expressions are critical for decision making controlling the logical flow of Python programs. The most fundamental way of doing this in Python is with a comparison operator, such as "`&lt;`":


```python
2 < 5
```

Python supplies serveral comparison operators:

<center>**Python Comparison Operators**</center>

| Operator |      Description      | Sample Input | Sample Output |
|:--------:|:---------------------:|:------------:|:-------------:|
| `&lt;`      | Less than             | `2 &lt; 5`      | `True`        |
| `&gt;`      | Greater than          | `2 &gt; 5`      | `False`       |
| `&lt;=`     | Less than or equal    | `2 &lt;= 5`     | `True`        |
|          |                       | `2 &lt;= 2`     | `True`        |
| `&gt;=`     | Greater than or equal | `2 &gt;= 5`     | `False`       |
| `==`     | Equality              | `2 == 2`     | `True`        |
|          |                       | `2 == 5`     | `False`       |
| `!=`     | Inequality            | `2 != 5`     | `True`        |
|          |                       | `2 != 2`     | `False`       |

Python does not restrict you to comparing just two operands at a time. For example:


```python
a, b, c = 1, 2, 3
a < b < c
```

This entire expression is `True` because `1 &lt; 2` is `True` and `2 &lt; 3` is `True`.

You can also use built-in functions in Python for comparing data. For example:


```python
min(3, 2.4, 5)
```


```python
max(3, 2.4, 5)
```

You can also combine comparison operators into compound expressions. For example:


```python
1 < 2 and 2 < 3
```

This compound expression returned `True` because **both** `1 &lt; 2` is true and `2 &lt; 3` is true. (Note that this is equivalent to `1 &lt; 2 &lt; 3`.)

### Exercise:


```python
# Now flip around one of the simple expressions and see if the output matches your expectations:
1 < 2 and 3 < 2
```

Python also provides the `or` Boolean operator, which requires that only one simple expression in a compound expression be true in order to return `True`. For example:


```python
1 < 2 or 1 > 2
```

Finally, `not` inverts the truth evaluation of an expression, such as in:


```python
not (2 < 3)
```

### Exercise:


```python
# Play around with compound expressions.
# Set i to different values to see what results this complex compound expression returns:
i = 7
(i == 2) or not (i % 2 != 0 and 1 < i < 5)
```

&gt; **Takeaway:** Arithmetic operations on numeric data form the foundation of data science work in Python. Even sophisticated numeric operations are predicated on these basics, so mastering them is essential to doing data science.

## Strings

&gt; **Learning goal:** By the end of this subsection, you should be comfortable working with strings at a basic level in Python.

Besides numbers, Python can also manipulate strings. Strings can be enclosed in single quotes (`'...'`) or double quotes (`"..."`) with the same result. Use `\` to escape quotes; that is, use `\` in order to use quotation marks within the string itself:


```python
'spam eggs'  # Single quotes.
```


```python
'doesn\'t'  # Use \' to escape the single quote...
```


```python
"doesn't"  # ...or use double quotes instead.
```

In the interactive interpreter and Jupyter Notebooks, the output string is enclosed in quotes and special characters are escaped with backslashes. Although this output sometimes looks different from the input (the enclosing quotes could change), the two strings are equivalent. The string is enclosed in double quotes if the string contains a single quote and no double quotes; otherwise, it’s enclosed in single quotes. The [`print()`](https://docs.python.org/3.6/library/functions.html#print) function produces a more readable output by omitting the enclosing quotes and by printing escaped and special characters:


```python
'"Isn\'t," she said.'
```


```python
print('"Isn\'t," she said.')
```

If you don't want escaped characters (prefaced by `\`) to be interpreted as special characters, use *raw strings* by adding an `r` before the first quote:


```python
print('C:\some\name')  # Here \n means newline!
```


```python
print(r'C:\some\name')  # Note the r before the quote.
```

### String literals

String literals can span multiple lines and are delineated by triple-quotes: `"""..."""` or `'''...'''`.

Because Python doesn't provide a means for creating multi-line comments, developers often just use triple quotes for this purpose. In a Jupyter notebook, however, such quotes define a string literal that appears as the output of a code cell:


```python
"""
Everything between the first three quotes, including new lines,
is part of the multi-line comment. Technically, the Python interpreter
simply sees the comment as a string, and because it's not otherwise
used in code, the string is ignored. Convenient, eh?
"""
```

For this reason, it's best in notebooks to use the # comment character at the beginning of each line, or better still, just use a Markdown cell outside of a code cell in a Jupyter notebook!

Strings can be *concatenated* (glued together) with the + operator, and repeated with *:


```python
# 3 times 'un', followed by 'ium'
3 * 'un' + 'ium'
```

The order of operations applies to operators when they are used with strings as well as numeric types. Try experimenting with different combinations and orders of operators and strings to see what happens.

### Concatenating strings

Two or more *string literals* placed next to each other are automatically concatenated:


```python
'Py' 'thon'
```

However, to concatenate variables or a variable and a literal, use `+`:


```python
prefix = 'Py'
prefix + 'thon'
```

### String indexes

Strings can be *indexed* (subscripted), with the first character having index 0. There is no separate character type; a character is simply a string of size one:


```python
word = 'Python'
word[0]  # Character in position 0.
```


```python
word[5]  # Character in position 5.
```

Indices may also be negative numbers, which means to start counting from the end of the string. Note that because -0 is the same as 0, negative indices start from -1:


```python
word[-1]  # Last character.
```


```python
word[-2]  # Second-last character.
```


```python
word[-6]
```

### Slicing strings

In addition to indexing, which extracts individual characters, Python also supports *slicing*, which extracts a substring. To slice, you indicate a *range* in the format `start:end`, where the start position is included but the end position is excluded:


```python
word[0:2]  # Characters from position 0 (included) to 2 (excluded).
```


```python
word[2:5]  # Characters from position 2 (included) to 5 (excluded).
```

If you omit either position, the default start position is 0 and the default end is the length of the string:


```python
word[:2]   # Character from the beginning to position 2 (excluded).
```


```python
word[4:]  # Characters from position 4 (included) to the end.
```


```python
word[-2:] # Characters from the second-last (included) to the end.
```

This characteristic means that `s[:i] + s[i:]` is always equal to `s`:


```python
word[:2] + word[2:]
```


```python
word[:4] + word[4:]
```

One way to remember how slices work is to think of the indices as pointing between characters, with the left edge of the first character numbered 0. Then the right edge of the last character of a string of *n* characters has index *n*. For example:
 +---+---+---+---+---+---+
 | P | y | t | h | o | n |
 +---+---+---+---+---+---+
 0   1   2   3   4   5   6
-6  -5  -4  -3  -2  -1
The first row of numbers gives the position of the indices 0–6 in the string; the second row gives the corresponding negative indices. The slice from *i* to *j* consists of all characters between the edges labeled *i* and *j*, respectively.

For non-negative indices, the length of a slice is the difference of the indices, if both are within bounds. For example, the length of `word[1:3]` is 2.

Attempting to use an index that is too large results in an error:


```python
word[42]  # The word only has 6 characters.
```

However, when used in a range, an index that's too large defaults to the size of the string and does not give an error. This characteristic is useful when you always want to slice at a particular index regardless of the length of a string:


```python
word[4:42]
```


```python
word[42:]
```

Python strings are [immutable](https://docs.python.org/3.6/glossary.html#term-immutable), which means they cannot be changed. Therefore, assigning a value to an indexed position in a string results in an error:


```python
word[0] = 'J'
```

The following cell also produces an error:


```python
word[2:] = 'py'
```

A slice is itself a value that you can concatenate with other values using `+`:


```python
'J' + word[1:]
```


```python
word[:2] + 'Py'
```

A slice, however, is not a string literal, and it cannot be used with automatic concatenation. The following code produces an error:


```python
word[:2] 'Py'    # Slice is not a literal; produces an error
```

Oftentimes, while working with strings, it can be useful to evaluate the length of a string. The built-in function [`len()`](https://docs.python.org/3.5/library/functions.html#len) returns the length of a string:


```python
s = 'supercalifragilisticexpialidocious'
len(s)
```

Another useful built-in function for working with strings is [`str()`](https://docs.python.org/3.6/library/stdtypes.html#str). This function takes any object and returns a printable string version of that object. For example:


```python
str(2)
```


```python
str(2.5)
```

&gt; **Takeaway:** Operations on string data form the other fundamental task you will do in data science in Python. Becoming comfortable with strings now will pay large dividends to you later as you work with increasingly complex data.

## Other data types

&gt; **Learning goal:** By the end of this subsection, you should have a basic understanding of the remaining fundamental data types in Python and an idea of how and when to use them.

The string and numeric data types that we have looked at so far are common to many programming languages. The other data types that we will now look at--lists, tuples, and dictionaries--set Python apart from C++ or Java by providing powerful and easy-to-use built-in data structures.

### Lists

Python knows a number of compound data types, which are used to group together other values. The most versatile is the [*list*](https://docs.python.org/3.5/library/stdtypes.html#typesseq-list), which can be written as a sequence of comma-separated values (items) between square brackets. Lists might contain items of different types, but usually the items all have the same type.


```python
squares = [1, 4, 9, 16, 25]
squares
```

Like strings (and all other built-in [sequence](https://docs.python.org/3.5/glossary.html#term-sequence) types), lists can be indexed and sliced:


```python
squares[0]  # Indexing returns the item.
```


```python
squares[-1]
```


```python
squares[-3:]  # Slicing returns a new list.
```

All slice operations return a new list containing the requested elements. This means that the following slice returns a new (shallow) copy of the list:


```python
squares[:]
```

Lists also support concatenation with the `+` operator:


```python
squares + [36, 49, 64, 81, 100]
```

Unlike strings, which are [immutable](https://docs.python.org/3.5/glossary.html#term-immutable), lists are a [mutable](https://docs.python.org/3.5/glossary.html#term-mutable) type, which means you can change any value in the list:


```python
cubes = [1, 8, 27, 65, 125]  # Something's wrong here ...
4 ** 3  # the cube of 4 is 64, not 65!
```


```python
cubes[3] = 64  # Replace the wrong value.
cubes
```

You can assign to slices, which can change the size of the list or clear it entirely:


```python
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
letters
```


```python
# Replace some values.
letters[2:5] = ['C', 'D', 'E']
letters
```


```python
# Now remove them.
letters[2:5] = []
letters
```


```python
# Clear the list by replacing all the elements with an empty list.
letters[:] = []
letters
```

The built-in [`len()`](https://docs.python.org/3.6/library/functions.html#len) function also applies to lists for getting their lengths:


```python
letters = ['a', 'b', 'c', 'd']
len(letters)
```

You can nest lists, which means to create lists that contain other lists. For example:


```python
a = ['a', 'b', 'c']
n = [1, 2, 3]
x = [a, n]
x
```

`x` is a list of lists, and you can access its constituent lists through the same indexing you use with simpler lists:


```python
x[0]
```

And by using additional index numbers, you can directly access elements within those sub-lists:


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

Python includes a number of handy functions that are available to all lists.

For example, [`append()`](https://docs.python.org/3.6/tutorial/datastructures.html) and [`extend()`](https://docs.python.org/3.6/tutorial/datastructures.html) enable you to add to the end of a list, much like the `+=` operator:


```python
beatles = ['John', 'Paul']
beatles.append('George')
beatles
```

Notice that you did not actually pass a list to `append()`; passing a list to `append()` results in this behavior:


```python
beatles2 = ['John', 'Paul', 'George']
beatles2.append(['Stuart', 'Pete'])
beatles2
```

To tack a list on the end of an existing list, use `extend()` instead:


```python
beatles.extend(['Stuart', 'Pete'])
beatles
```

[`index()`](https://docs.python.org/3.6/tutorial/datastructures.html) returns the index of the first matching item in a list (if present):


```python
beatles.index('George')
```

The [`count()`](https://docs.python.org/3.6/tutorial/datastructures.html) method returns the number of items in a list that match objects you pass in:


```python
beatles.count('John')
```

There are two methods for removing items from a list. The first is [`remove()`](https://docs.python.org/3.6/tutorial/datastructures.html), which locates the first occurrence of an item in the list and removes it (if present):


```python
beatles.remove('Stuart')
beatles
```

The other method for removing items from lists is the [`pop()`](https://docs.python.org/3.6/tutorial/datastructures.html) method. If you supply `pop()` with an index number, it will remove the item from that location in the list and return it; otherwise, `pop()` removes the last item in a list and returns that:


```python
beatles.pop()
```

The [`insert()`](https://docs.python.org/3.6/tutorial/datastructures.html) method enables you to add an item to a specific location in a list:


```python
beatles.insert(1, 'Ringo')
beatles
```

Unsurprisingly, the [`reverse()`](https://docs.python.org/3.6/tutorial/datastructures.html) method reverses the order of items in a list:


```python
beatles.reverse()
beatles
```

Finally, the [`sort()`](https://docs.python.org/3.6/tutorial/datastructures.html) method orders the items in a list:


```python
beatles.sort()
beatles
```

### Exercise:


```python
# What happens if you run beatles.extend(beatles)?
# How about beatles.append(beatles)?

```

Note that you can supply your own *lambda function* to `sort()` for use in comparing items in a list. We will cover lambda functions in Section 2.

### Tuples

Another immutable data type in Python are *tuples*. It can be useful at times to create a data structure that won't be altered later in a program, such as to protect constant data from being overwritten on accident or to improve performance for iterating over data. This is where tuples come in. You create tuples much as you do lists, only using parentheses instead of brackets.


```python
t = (1, 2, 3)
t
```

Because tuples are immutable, you cannot change elements within them:


```python
t[1] = 2.0
```

However, you can refer to elements within them:


```python
t[1]
```

You can also slice tuples:


```python
t[:2]
```

You can also create tuples from lists:


```python
l = ['baked', 'beans', 'spam']
l = tuple(l)
l
```

Or you can create lists from tuples:


```python
l = list(l)
l
```

### Membership testing

As your Python programming grows more complex, you will want to test lists and tuples for the membership of specific data. The `in` operator enables you to do that.


```python
tup = ('a', 'b', 'c')
'b' in tup
```

You can also test to see if something is not in a list or tuple using `not in`:


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

Dictionaries in Python provide a means of mapping information between unique keys and values. You create dictionaries by listing zero or more key-value pairs inside of braces, like this:


```python
capitals = {'France': ('Paris', 2140526)}
```

Keys for dictionaries can be three things: strings, numbers, or tuples (that contain only strings, numbers, or other tuples). The important thing is that dictionary keys be immutable, so lists cannot be used for keys in dictionaries, for example.

You add to dictionaries like this:


```python
capitals['Nigeria'] = ('Lagos', 6048430)
capitals
```

### Exercise:


```python
# Now try adding another country (or something else) to the capitals dictionary
```

You reference entries much like you do as through an index number for a string, list, or tuple, but instead of an index, use a key:


```python
capitals['France']
```

You can also update entries in the dictionary:


```python
capitals['Nigeria'] = ('Abuja', 1235880)
capitals
```

When used on a dictionary, the `len()` method returns the number of keys in a dictionary:


```python
len(capitals)
```

Similar to the `pop()` method for lists, the `popitem()` method randomly removes a key from the dictionary, along with its associated value:


```python
capitals.popitem()
```


```python
capitals
```

&gt; **Takeaway:** Regardless of how complex and voluminous the data you will work with, these basic data structures will repeatedly be your means for handling and manipulating it. Comfort with these basic data structures is essential to being able to understand and use Python code written by others.

## Control Flow, Functions, and Libraries

You will have a foundational understanding of Python control flows, functions, and libraries after completing this section.

### Control flow in Python

&gt; **Learning goal:** By the end of this subsection, you should be comfortable using basic control flows in Python.

Now that you have a working understanding of the fundamental data types and structures in Python, we can move on to actual programming using Python.

#### If-statements

`If` statements in Python are similar to those in other programming languages like Java, and they form the backbone of the logical flow of most programs.


```python
y = 6
if y % 2 == 0:
    print('Even')
```

### Exercise:


```python
# What behavior do you experience if you change y to be odd?
```

Did you notice the indentation for print under the if statement? That indentation is important because that is how Python demarks the scope of a control flow--what is contingently run or looped over--as opposed to the braces ({}) used in other languages.

To cover more contingencies without having to construct a follow-on `if` statement, you can add an `else` statement:


```python
y = 7
if y % 2 == 0:
    print('Even')
else:
    print('Odd')
```

`elif` enables you to insert an additional logical test to an `if` statement:


```python
y = 1
if y % 2 == 0:
    print('Even')
elif y == 1:
    print('One')
else:
    print('Odd')
```

Notice that, in the previous example, the `if` statement exited after finding the *first* logical test that was `true`. If `y = 1`, and while 1 is indeed odd, the `if` statement executed and exited after finding that `y == 1`, rather than continuing to the end of the statement.

### Exercise:


```python
# Try changing the value of y in the snippet above.
# Do you get the output that you expect?

```

#### For-loops

It is often necessary in programs to iterate over some set of items. This is where `for` loops prove useful. For example, they can provide a useful way to iterate over the items of a list:


```python
colors = ['red', 'yellow', 'blue']
for color in colors:
    print(color)
```

Sometimes, you will want to iterate over a list using the list index rather than items from that list (say, when you want to access items from another list at the same time). In this case, you can combine list-object methods and for loops:


```python
comp_colors = ['green', 'purple', 'orange']
for i in range(len(comp_colors)):
    print(colors[i], comp_colors[i])
```

We've met `len()` before, but [`range()`](https://docs.python.org/3/library/functions.html#func-range) is new to us. That function produces a sequence of integers from 0 to 1 less than the number passed into it. Hence:


```python
for j in range(5):
    print(j)
```

In addition to `range(`*`stop`*`)`, the range function can take up to three parameters: `range(`*`start`*, *`stop`*`[, `*step*`])`. This odd-looking notation just means that if you pass a single argument to `range()`, it will take that to be the stop value; two arguments will be the start and stop values; and three values are `start`, `stop`, and `step`.

### Exercise:


```python
# How would you use range and a for loop to print the sequence of numbers
# from 10 to 20? How about counting by threes from 17 to 41?

```

It can also be important to break out of a loop. Python uses the `break` statement borrowed from C to do this. To see this in action, consider two nested for loops:


```python
for n in range(2, 10):
    for x in range(2, n):
        if n % x == 0:
            print(n, 'equals', x, '*', n//x)
            break
    else:
        print(n, 'is a prime number')
```

Note that, in the example above, the `else` statement belongs to the `for` loop, not to the `if` statement.

### Exercise:


```python
# Try changing the code snippet above after you remove the break statement.
# What output does it now produce?

```

As part of the control flow of your program, you might want to continue to the next iteration of your loop. The `continue` statement (also borrowed from C) can help with that:


```python
for num in range(2, 10):
    if num % 2 == 0:
        print("Found an even number:", num)
        continue
    print("Found an odd number:", num)
```

### Exercise:


```python
# What happens when you replace the continue statement above with a break?

```

#### While-loops

If we cross the functionality of the `if` statement with that of the `for` loop, we would get the `while` loop, a loop that iterates while some logical condition remains true. Consider this snippet of code to compute the initial sub-sequence of the Fibonacci sequence:


```python
# In the Fibonacci series, the sum of two elements defines the next.
a, b = 0, 1

while b < 100:    
    print(b, end=', ')
    a, b = b, a+b
```

Go ahead and play with the number of iterations for the while loop. Notice that this snippet also uses multiple assignment for variables.

&gt; **Takeaway:** Control flows are what make programs programs, as opposed to a single sequence of operations. Mastering the logical flow of information in Python will enable you to automate tasks that would be impossibly complex or time-consuming to do manually.

### Functions

&gt; **Learning goal:** By the end of this subsection, you should understand how to pass and receive data from functions.

As in other programming languages, it is often essential in Python to break down your program into reusable chunks. A primary means of doing that is through functions.

For example, we could rewrite the `while` loop code snippet above as a formal function:


```python
def fib(n):
    """Print a Fibonacci series up to n."""
    a, b = 0, 1
    while a < n:
        print(a, end=', ')
        a, b = b, a+b
```

Now we can call this function and compute the Fibonacci series up to some arbitrary point:


```python
fib(2000)
```

Python can also define new functions on the fly. These anonymous functions are called *lambda functions* because you define them with the `lambda` keyword. Lambda functions can contain any number of arguments but only one expression.


```python
nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
list(filter(lambda x: x % 2 != 0, nums))
```

&gt; **Takeaway:** You will constantly be using functions of all kind to perform data science in Python, so understanding how functions accept, work on, and return data is critical to further progress.

### List comprehensions

&gt; **Learning goal:** By the end of this subsection, you should understand how to economically and computationally create lists.

Sometimes, it makes more sense to generate a list algorithmically. Consider the last example. We really wanted just a list of numbers from 1 to 10. Rather than type those out, we can use a *list comprehension* to generate it:


```python
numbers = [x for x in range(1,11)] # Remember to create a range 1 more than the number you actually want.
numbers
```

We can also perform computation on the items generated for the list:


```python
squares = [x*x for x in range(1,11)]
squares
```

We can even perform logical tests on list items in the comprehension:


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

If you quit from the Python interpreter and enter it again, the definitions you have made (your functions and variables) will be lost. Similarly, you might also want to use a handy function that you’ve written in several programs without copying its definition into each program.

To support this, Python has a way to put definitions in a file and use them in a script or in an interactive instance of the interpreter. Such a file is called a [*module*](https://docs.python.org/3/tutorial/modules.html). Definitions from a module can be imported into other programs or modules.

For example, the `factorial()` function is not one of the standard functions built into Python. It is part of the Python [`math`](https://docs.python.org/3/library/math.html) module. So, when we run `factorial()` before importing `math`, we get an error:


```python
factorial(5)
```

However, the situation changes after we import the `math` module:


```python
import math
math.factorial(5)
```

Notice that we still have to prepend `math` to the front of the `factorial()` function. We can use a different method to import that specific function from the `math` module and use it as if it were defined in our program:


```python
from math import factorial
factorial(5)
```

You can add more cells to your notebook by clicking the **insert cell below (+)** button at the top of the window. The Python [`math`](https://docs.python.org/3/library/math.html) module has many functions in it. Try importing some of the other math functions and playing around with them.

&gt; **Takeaway:** There are several Python modules that you will regularly use in conducting data science in Python, so understanding how to import them will be essential (especially in this training).
