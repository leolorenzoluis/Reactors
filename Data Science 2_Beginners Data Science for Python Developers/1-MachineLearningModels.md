
# Section 1: Introduction to machine learning models 


## A quick aside: types of ML

As you get deeper into data science, it might seem like there are a bewildering array of ML algorithms out there. However many you encounter, it can be handy to remember that most ML algorithms fall into three broad categories:
 - **Predictive algorithms**: These analyze current and historical facts to make predictions about unknown events, such as the future or customers’ choices.
 - **Classification algorithms**: These teach a program from a body of data, and the program then uses that learning to classify new observations.
 - **Time-series forecasting algorithms**: While it can argued that these algorithms are a part of predictive algorithms, their techniques are specialized enough that they in many ways functions like a separate category. Time-series forecasting is beyond the scope of this course, but we have more than enough work with focusing here on prediction and classification.

## Prediction: linear regression

&gt; **Learning goal:** By the end of this subsection, you should be comfortable fitting linear regression models, and you should have some familiarity with interpreting their output.

### Data exploration

**Import Libraries**


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 
import seaborn as sns
```

**Dataset Alert**: Boston Housing Dataset


```python
df = pd.read_csv('Data/Housing_Dataset_Sample.csv')
df.head()
```

### Exercise:


```python
# Do you remember the DataFrame method for looking at overall information
# about a DataFrame, such as number of columns and rows? Try it here.

```


```python
df.describe().T
```

**Price Column**


```python
sns.distplot(df['Price'])
```

**House Prices vs Average Area Income**


```python
sns.jointplot(df['Avg. Area Income'],df['Price'])
```

**All Columns**


```python
sns.pairplot(df)
```

**Some observations**  
1. Blob Data
2. Distortions might be a result of data (e.g. no one has 0.3 rooms)


### Fitting the model

**Can We Predict Housing Prices?**


```python
X = df.iloc[:,:5] # First 5 Columns
y = df['Price']   # Price Column
```

**Train, Test, Split**


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=54)
```

**Fit to Linear Regression Model**


```python
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
```


```python
reg.fit(X_train,y_train)
```

### Evaluating the model

**Predict**


```python
predictions = reg.predict(X_test)
```


```python
predictions
```


```python
print(reg.intercept_,reg.coef_)
```

**Score**


```python
#Explained variation. A high R2 close to 1 indicates better prediction with less error.
from sklearn.metrics import r2_score

r2_score(y_test,predictions)
```

**Visualize Errors**


```python
sns.distplot([y_test-predictions])
```

**Visualize Predictions**


```python
# Plot outputs
plt.scatter(y_test,predictions, color='blue')
```

### Exercise:
Can you think of a way to refine this visualization to make it clearer, particularly if you were explaining the results to someone?


```python
# Hint: Remember to try the plt.scatter parameter alpha=.
# It takes values between 0 and 1.

```

&gt; **Takeaway:** In this subsection, you performed prediction using linear regression by exploring your data, then fitting your model, and finally evaluating your model’s performance.

## Classification: logistic regression

&gt; **Learning goal:** By the end of this subsection, you should know how logistic regression differs from linear regression, be comfortable fitting logistic regression models, and have some familiarity with interpreting their output.

**Dataset Alert**: Fates of RMS Titanic Passengers

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
df = pd.read_csv('Data/train_data_titanic.csv')
df.head()
```


```python
df.info()
```



### Remove extraneous variables


```python
df.drop(['Name','Ticket'],axis=1,inplace=True)
```



### Check for multicollinearity

**Question**: Do any correlations between **Survived** and **Fare** jump out?


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


```python
df.head()
```


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



```python
# missing
df.isnull().sum()>(len(df)/2)
```

    The history saving thread hit an unexpected error (OperationalError('no such table: history',)).History will not be written to the database.



    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-1-a63f128a4173> in <module>()
          1 # missing
    ----> 2 df.isnull().sum()>(len(df)/2)
    

    NameError: name 'df' is not defined



```python
df.drop('Cabin',axis=1,inplace=True)
```


```python
df.info()
```


```python
df['Age'].isnull().value_counts()
```

### Corelation Exploration


```python
df.groupby('Sex')['Age'].median().plot(kind='bar')
```


```python
df['Age'] = df.groupby('Sex')['Age'].apply(lambda x: x.fillna(x.median()))
```


```python
df.isnull().sum()
```


```python
df['Embarked'].value_counts()
```


```python
df['Embarked'].fillna(df['Embarked'].value_counts().idxmax(), inplace=True)
df['Embarked'].value_counts()
```


```python
df = pd.get_dummies(data=df, columns=['Sex', 'Embarked'],drop_first=True)
df.head()
```

**Correlation Matrix**


```python
df.corr()
```

**Define X and Y**


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

**Use Logistic Regression Model**


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


#### Classification report


```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
```

The classification reports the proportions of both survivors and non-survivors with four scores:
 - **Precision:** The number of true positives divided by the sum of true positives and false positives; closer to 1 is better.
 - **Recall:** The true-positive rate, the number of true positives divided by the sum of the true positives and the false negatives.
 - **F1 score:** The harmonic mean (the average for rates) of precision and recall.
 - **Support:** The number of true instances for each label.


```python
print(classification_report(y_test,predictions))
```

#### Confusion matrix


```python
print(confusion_matrix(y_test,predictions))
```


```python
pd.DataFrame(confusion_matrix(y_test, predictions), columns=['True Survived', 'True Not Survived'], index=['Predicted Survived', 'Predicted Not Survived'])
```

#### Accuracy score


```python
print(accuracy_score(y_test,predictions))
```

&gt; **Takeaway:** In this subsection, you performed classification using logistic regression by removing extraneous variables, checking for multicollinearity, handling missing values, and fitting and evaluating your model.

## Classification: decision trees

&gt; **Learning goal:** By the end of this subsection, you should be comfortable fitting decision-tree models and have some understanding of what they output.


```python
from sklearn import tree
tr = tree.DecisionTreeClassifier()
```

### Exercise:


```python
# Using the same split data as with the logistic regression,
# can you fit the decision tree model?
# Hint: Refer to code snippet for fitting the logistic regression above.

```

**Note**: Using the same Titanic Data


```python
tr.fit(X_train, y_train)
```


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

**Visualize tree**


```python
import graphviz 

dot_file = tree.export_graphviz(tr, out_file=None, 
                                feature_names=X.columns, 
                                class_names='Survived',  
                                filled=True,rounded=True)  
graph = graphviz.Source(dot_file)  
graph
```

&gt; **Takeaway:** In this subsection, you performed classification on previously cleaned data by fitting and evaluating a decision tree.
