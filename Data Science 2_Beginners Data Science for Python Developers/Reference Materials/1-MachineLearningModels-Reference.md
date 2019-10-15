
# Section 1: Introduction to machine learning models 

You have now made it to the section on machine learning (ML). ML and the branch of computer science in which it resides, artificial intelligence (AI), are so central to data science that ML/AI and data science are synonymous in the minds of many people. However, the preceding sections have hopefully demonstrated that there are a lot of other facets to the discipline of data science apart from the prediction and classification tasks that supply so much value to the world. (Remember, at least 80 percent of the effort in most data-science projects will be composed of cleaning and manipulating the data to prepare it for analysis.)

That said, ML is fun! In this section, and the next one on data science in the cloud, you will get to play around with some of the “magic” of data science and start to put into practice the tools you have spent the last five sections learning. Let's get started!

## A quick aside: types of ML

As you get deeper into data science, it might seem like there are a bewildering array of ML algorithms out there. However many you encounter, it can be handy to remember that most ML algorithms fall into three broad categories:
 - **Predictive algorithms**: These analyze current and historical facts to make predictions about unknown events, such as the future or customers’ choices.
 - **Classification algorithms**: These teach a program from a body of data, and the program then uses that learning to classify new observations.
 - **Time-series forecasting algorithms**: While it can argued that these algorithms are a part of predictive algorithms, their techniques are specialized enough that they in many ways functions like a separate category. Time-series forecasting is beyond the scope of this course, but we have more than enough work with focusing here on prediction and classification.

## Prediction: linear regression

&gt; **Learning goal:** By the end of this subsection, you should be comfortable fitting linear regression models, and you should have some familiarity with interpreting their output.

Arguably the simplest form of machine learning is to draw a line connecting two points and make predictions about where that trend might lead.

But what if you have more than two points—and those points don't line up neatly? What if you have points in more than two dimensions? This is where linear regression comes in.

Formally, linear regression is used to predict a quantitative *response* (the values on a Y axis) that is dependent on one or more *predictors* (values on one or more axes that are orthogonal to Y, commonly just thought of collectively as X). The working assumption is that the relationship between predictors and response is more or less linear. The goal of linear regression is to fit a straight line in the best possible way to minimize the deviation between our observed responses in the dataset and the responses predicted by our line, the linear approximation. (The most common means of assessing this error is called the **least squares method**; it consists of minimizing the number you get when you square the difference between your predicted value and the actual value and add up all of those squared differences for your entire dataset.)

<img src="../Images/linear_regression.png" style="padding-right: 10px;">


Statistically, we can represent this relationship between response and predictors as:

$Y = B_0 + B_1X + E$

Remember high school geometry? $B_0$ is the intercept of our line and $B_1$ is its slope. We commonly refer to $B_0$ and $B_1$ as coefficients and to $E$ as the *error term*, which represents the margin of error in the model.

Let's try this in practice with actual data. (Note: no graph paper will be harmed in the course of these predictions.)

### Data exploration

We'll begin by importing our usual libraries and using our %matplotlib inline magic command:


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 
import seaborn as sns
```

And now for our data. In this case, we’ll use a newer housing dataset than the Boston Housing Dataset we used in the last section (with this one storing data on individual houses across the United States).


```python
df = pd.read_csv('../Data/Housing_Dataset_Sample.csv')
df.head()
```

### Exercise:


```python
# Do you remember the DataFrame method for looking at overall information
# about a DataFrame, such as number of columns and rows? Try it here.

```

Let's also use the `describe` method to look at some of the vital statistics about the columns. Note that in cases like this, in which some of the column names are long, it can be helpful to view the transposition of the summary, like so:


```python
df.describe().T
```

Let's look at the data in the **Price** column. (You can disregard the deprecation warning if it appears.)


```python
sns.distplot(df['Price'])
```

As we would hope with this much data, our prices form a nice bell-shaped, normally distributed curve.

Now, let's look at a simple relationship like that between house prices and the average income in a geographic area:


```python
sns.jointplot(df['Avg. Area Income'],df['Price'])
```

As we would expect, there is an intuitive, linear relationship between them. Also good: the pairplot shows that the data in both columns is normally distributed, so we don't have to worry about somehow transforming the data for meaningful analysis.

Let's take a quick look at all of the columns:


```python
sns.pairplot(df)
```

Some observations:
1. Not all of the combinations of columns provide strong linear relationships; some just look like blobs. That's nothing to worry about for our analysis.
2. See the visualizations that look like lanes rather than organic groups? That is the result of the average number of bedrooms in houses being measured in discrete values rather than continuous ones (as no one has 0.3 bedrooms in their house). The number of bathrooms is also the one column whose data is not really normally distributed, though some of this might be distortion caused by the default bin size of the pairplot histogram functionality.

It is now time to make a prediction. 

### Fitting the model

Let's make a prediction. Let's feed everything into a linear model (average area income, average area house age, average area number of rooms, average area number of bedrooms, and	area population) and see how well knowing those factors can help us predict the price of a home. 

To do this, we will make our first five columns the X (our predictors) and the **Price** column the Y (our response):


```python
X = df.iloc[:,:5]
y = df['Price']
```

Now, we could use all of our data to create our model. However, all that would get us is a model that is good at predicting itself. Not only would that leave us with no objective way to measure how good the model is, it would also likely lead to a model that was less accurate when used on new data. Such a model is termed *overfitted*.

To avoid this, data scientists divide their datasets for ML into *training* data (the data used to fit the model) and *test* data (data used to evaluate how accurate the model is). Fortunately, scikit-learn provides a function that enables us to easily divide up our data between training and test sets: `train_test_split`. In this case, we will use 70 percent of our data for training and reserve 30 percent of it for testing. (Note that you will also supply a fourth parameter to the function: `random_state`; `train_test_split` randomly divides up our data between test and training, so this number provides an explicit seed for the random-number generator so that you will get the same result each time you run this code snippet.)


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=54)
```

All that is left now is to import our linear regression algorithm and fit our model based on our training data:


```python
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
```


```python
reg.fit(X_train,y_train)
```

### Evaluating the model

Now, a moment of truth: let's see how our model does making predictions based on the test data:


```python
predictions = reg.predict(X_test)
```


```python
predictions
```

Our predictions are just an array of numbers: these are the house prices predicted by our model. One for every row in our test dataset.

Remember how we mentioned that linear models have the mathematical form of $Y = B_0 + B_1*X + E$? Let’s look at the actual equation:


```python
print(reg.intercept_,reg.coef_)
```

In algebraic terms, here is our model:

$Y=-2,646,401+0.21587X_1+0.00002X_2+0.00001X_3+0.00279X_4+0.00002X_5$

where:
 - $Y=$ Price
 - $X_1=$ Average area income
 - $X_2=$ Average area house age
 - $X_3=$ Average area number of rooms
 - $X_4=$ Average area number of bedrooms
 - $X_5=$ Area population

So, just how good is our model? There are many ways to measure the accuracy of ML models. Linear models have a good one: the $R^2$ score (also knows as the coefficient of determination). A high $R^2$, close to 1, indicates better prediction with less error.


```python
#Explained variation. A high R2 close to 1 indicates better prediction with less error.
from sklearn.metrics import r2_score

r2_score(y_test,predictions)
```

The $R^2$ score also indicates how much explanatory power a linear model has. In the case of our model, the five predictors we used in the model explain a little more than 92 percent of the price of a house in this dataset.

We can also plot our errors to get a visual sense of how wrong our predictions were:


```python
#plot errors
sns.distplot([y_test-predictions])
```

Do you notice the numbers on the left axis? Whereas a histogram shows the number of things that fall into discrete numeric buckets, a kernel density estimation (KDE, and the histogram that accompanies it in the Seaborn displot) normalizes those numbers to show what proportion of results lands in each bucket. Essentially, these are all decimal numbers less than 1.0 because the area under the KDE has to add up to 1.

Maybe more gratifying, we can plot the predictions from our model:


```python
# Plot outputs
plt.scatter(y_test,predictions, color='blue')
```

The linear nature of our predicted prices is clear enough, but there are so many of them that it is hard to tell where dots are concentrated. Can you think of a way to refine this visualization to make it clearer, particularly if you were explaining the results to someone?

### Exercise:


```python
# Hint: Remember to try the plt.scatter parameter alpha=.
# It takes values between 0 and 1.

```

&gt; **Takeaway:** In this subsection, you performed prediction using linear regression by exploring your data, then fitting your model, and finally evaluating your model’s performance.

## Classification: logistic regression

&gt; **Learning goal:** By the end of this subsection, you should know how logistic regression differs from linear regression, be comfortable fitting logistic regression models, and have some familiarity with interpreting their output.

We'll now pivot to discussing classification. If our simple analogy of predictive analytics was drawing a line through points and extrapolating from that, then classification can be described in its simplest form as drawing lines around groups of points.

While linear regression is used to predict quantitative responses, *logistic* regression is used for classification problems. Formally, logistic regression predicts the categorical response (Y) based on predictors (Xs). Logistic regression goes by several names, and it is also known in the scholarly literature as logit regression, maximum-entropy classification (MaxEnt), and the log-linear classifier. In this algorithm, the probabilities describing the possible outcomes of a single trial are modeled using a sigmoid (S-curve) function. Sigmoid functions take any value and transform it to be between 0 and 1, which can be used as a probability for a class to be predicted, with the goal of predictors mapping to 1 when something belongs in the class and 0 when they do not.

<img src="../Images/logistic_regression.png?" style="padding-right: 10px;">

To show this in action, let's do something a little different and try a historical dataset: the fates of the passengers of the RMS Titanic, which is a popular dataset for classification problems in machine learning. In this case, the class we want to predict is whether a passenger survived the doomed liner's sinking.

The dataset has 12 variables:

 - **PassengerId**
 - **Survived:** 0 = No, 1 = Yes
 - **Pclass:** Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
 - **Sex**
 - **Age**		
 - **Sibsp:** Number of siblings or spouses aboard the *Titanic*	
 - **Parch:** Number of parents or children aboard the *Titanic*
 - **Ticket:** Passenger ticket number	
 - **Fare:** Passenger fare	
 - **Cabin:** Cabin number	
 - **Embarked:** Port of embarkation; C = Cherbourg, Q = Queenstown, S = Southampton


```python
df = pd.read_csv('../Data/train_data_titanic.csv')
df.head()
```


```python
df.info()
```

One reason that the Titanic data set is a popular classification set is that it provides opportunities to prepare data for analysis. To prepare this dataset for analysis, we need to perform a number of tasks:
 - Remove extraneous variables
 - Check for multicollinearity 
 - Handle missing values

We will touch on each of these steps in turn.

### Remove extraneous variables

The name of individual passengers and their ticket numbers will clearly do nothing to help our model, so we can drop those columns to simplify matters.


```python
df.drop(['Name','Ticket'],axis=1,inplace=True)
```

There are additional variables that will not add classifying power to our model, but to find them we will need to look for correlation between variables.

### Check for multicollinearity

If one or more of our predictors can themselves be predicted from other predictors, it can produce a state of *multicollinearity* in our model. Multicollinearity is a challenge because it can skew the results of regression models (both linear and logistic) and reduce the predictive or classifying power of a model.

To help combat this problem, we can start to look for some initial patterns. For example, do any correlations between **Survived** and **Fare** jump out?


```python
sns.pairplot(df[['Survived','Fare']], dropna=True)
```

### Exercise:


```python
# Try running sns.pairplot twice more on some other combinations of columns
# and see if any patterns emerge.

```

We can also use `groupby` to look for patterns. Consider the mean values for the various variables when we group by **Survived**:


```python
df.groupby('Survived').mean()
```

Survivors appear to be slightly younger on average with higher-cost fare.


```python
df.head()
```

Value counts can also help us get a sense of the data before us, such as numbers for siblings and spouses on the *Titanic*, in addition to the sex split of passengers:


```python
df['SibSp'].value_counts()
```


```python
df['Parch'].value_counts()
```


```python
df['Sex'].value_counts()
```

### Handle missing values

We now need to address missing values. First, let’s look to see which columns have more than half of their values missing:


```python
#missing
df.isnull().sum()>(len(df)/2)
```

Let's break down the code in the call above just a bit. `df.isnull().sum()` tells pandas to take the sum of all of the missing values for each column. `len(df)/2` is just another way of expressing half the number of rows in the `DataFrame`. Taken together with the `&gt;`, this line of code is looking for any columns with more than half of its entries missing, and there is one: **Cabin**.

We could try to do something about those missing values. However, if any pattern does emerge in the data that involves **Cabin**, it will be highly cross-correlated with both **Pclass** and **Fare** (as higher-fare, better-class accommodations were grouped together on the *Titanic*). Given that too much cross-correlation can be detrimental to a model, it is probably just better for us to drop **Cabin** from our `DataFrame`:


```python
df.drop('Cabin',axis=1,inplace=True)
```

Let's now run `info` to see if there are columns with just a few null values.


```python
df.info()
```

One note on the data: given that 1,503 died in the *Titanic* tragedy (and that we know that some survived), this data set clearly does not include every passenger on the ship (and none of the crew). Also remember that **Survived** is a variable that includes both those who survived and those who perished.

Back to missing values. **Age** is missing several values, as is **Embarked**. Let's see how many values are missing from **Age**:


```python
df['Age'].isnull().value_counts()
```

As we saw above, **Age** isn't really correlated with **Fare**, so it is a variable that we want to eventually use in our model. That means that we need to do something with those missing values. But we before we decide on a strategy, we should check to see if our median age is the same for both sexes.


```python
df.groupby('Sex')['Age'].median().plot(kind='bar')
```

The median ages are different for men and women sailing on the *Titanic*, which means that we should handle the missing values accordingly. A sound strategy is to replace the missing ages for passengers with the median age *for the passengers' sexes*.


```python
df['Age'] = df.groupby('Sex')['Age'].apply(lambda x: x.fillna(x.median()))
```

Any other missing values?


```python
df.isnull().sum()
```

We are missing two values for **Embarked**. Check to see how that variable breaks down:


```python
df['Embarked'].value_counts()
```

The vast majority of passengers embarked on the *Titanic* from Southampton, so we will just fill in those two missing values with the most statistically likely value (the median result): Southampton.


```python
df['Embarked'].fillna(df['Embarked'].value_counts().idxmax(), inplace=True)
df['Embarked'].value_counts()
```


```python
df = pd.get_dummies(data=df, columns=['Sex', 'Embarked'],drop_first=True)
df.head()
```

Let's do a final look at the correlation matrix to see if there is anything else we should remove.


```python
df.corr()
```

**Pclass** and **Fare** have some amount of correlation, we can probably get rid of one of them. In addition, we need to remove **Survived** from our X `DataFrame` because it will be our response `DataFrame`, Y:


```python
X = df.drop(['Survived','Pclass'],axis=1)
y = df['Survived']
```


```python
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=67)
```

### Exercise:

We now need to split the training and test data, which you will so as an exercise:


```python
from sklearn.model_selection import train_test_split
# Look up in the portion above on linear regression and use train_test_split here.
# Set test_size = 0.3 and random_state = 67 to get the same results as below when
# you run through the rest of the code example below.

```

Now you will import and fit the logistic regression model:


```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
```


```python
lr.fit(X_train,y_train)
```


```python
predictions = lr.predict(X_test)
```

### Evaluate the model

In contrast to linear regression, logistic regression does not produce an $R^2$ score by which we can assess the accuracy of our model. In order to evaluate that, we will use a classification report, a confusion matrix, and the accuracy score.

#### Classification report


```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
```

The classification reports the proportions of both survivors and non-survivors with four scores:
 - **Precision:** The number of true positives divided by the sum of true positives and false positives; closer to 1 is better.
 - **Recall:** The true-positive rate, the number of true positives divided by the sum of the true positives and the false negatives.
 - **F1 score:** The harmonic mean (the average for rates) of precision and recall.
 - **Support:** The number of true instances for each label.
 
Why so many ways of measuring accuracy for a model? Well, success means different things in different contexts. Imagine that we had a model to diagnose infectious disease. In such a case we might want to tune our model to maximize recall (and thus minimize our false-negative rate): even high precision might miss a lot of infected people. On the other hand, a weather-forecasting model might be interested in maximizing precision because the cost of false negatives is so low. For other uses, striking a balance between precision and recall by maximizing the F1 score might be the best choice. Run the classification report:


```python
print(classification_report(y_test,predictions))
```

#### Confusion matrix

The confusion matrix is another way to present this same information, this time with raw scores. The columns show the true condition, positive on the left, negative on the right. The rows show predicted conditions, positive on the top, negative on the bottom. So, the matrix below shows that our model correctly predicted 146 survivors (true positives) and incorrectly predicted another 16 (false positives). On the other hand, our model correctly predicted 30 non-survivors (true negatives) and incorrectly predicted 76 more (false negatives).


```python
print(confusion_matrix(y_test,predictions))
```

Let's dress up the confusion matrix a bit to make it a little easier to read:


```python
pd.DataFrame(confusion_matrix(y_test, predictions), columns=['True Survived', 'True Not Survived'], index=['Predicted Survived', 'Predicted Not Survived'])
```

#### Accuracy score

Finally, our accuracy score tells us the fraction of correctly classified samples; in this case (146 + 76) / (146 + 76 + 30 + 16).


```python
print(accuracy_score(y_test,predictions))
```

Not bad for an off-the-shelf model with no tuning!

&gt; **Takeaway:** In this subsection, you performed classification using logistic regression by removing extraneous variables, checking for multicollinearity, handling missing values, and fitting and evaluating your model.

## Classification: decision trees

&gt; **Learning goal:** By the end of this subsection, you should be comfortable fitting decision-tree models and have some understanding of what they output.

If logistic regression uses observations about variables to swing a metaphorical needle between 0 and 1, classification based on decision trees programmatically builds a Yes/No decision to classify items.

<img src="../Images/decision_tree.png" style="padding-right: 10px;">

Let's look at this in practice with the same *Titanic* dataset we used with logistic regression.


```python
from sklearn import tree
```


```python
tr = tree.DecisionTreeClassifier()
```

### Exercise:


```python
# Using the same split data as with the logistic regression,
# can you fit the decision tree model?
# Hint: Refer to code snippet for fitting the logistic regression above.

```


```python
tr.fit(X_train, y_train)
```

Once fitted, we get our predicitions just like we did in the logistic regression example above:


```python
tr_predictions = tr.predict(X_test)
```


```python
pd.DataFrame(confusion_matrix(y_test, tr_predictions), 
             columns=['True Survived', 'True Not Survived'], 
             index=['Predicted Survived', 'Predicted Not Survived'])
```


```python
print(accuracy_score(y_test,tr_predictions))
```

One of the great attractions of decision trees is that the models are readable by humans. Let's visualize to see it in action. (Note, the generated graphic can be quite large, so scroll to the right if the generated graphic just looks blank at first.)


```python
import graphviz 

dot_file = tree.export_graphviz(tr, out_file=None, 
                                feature_names=X.columns, 
                                class_names='Survived',  
                                filled=True,rounded=True)  
graph = graphviz.Source(dot_file)  
graph
```

There are, of course, myriad other ML models that we could explore. However, you now know some of the most commonly encountered ones, which is great preparation to understand what automated, cloud-based ML and AI services are doing and how to intelligently apply them to data-science problems, the subject of the next section.

&gt; **Takeaway:** In this subsection, you performed classification on previously cleaned data by fitting and evaluating a decision tree.
