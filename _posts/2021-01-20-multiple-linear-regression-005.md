---
layout: post
title: "Multiple Linear Regression #5"
subtitle: Using scikit-learn, statsmodels, seaborn, and plotly
comments: false
---

This is the fifth post in a series on the multiple linear regression model.  In previous posts, I introduced the theory behind the model, exploring using Python's scikit-learn and statsmodels libraries, and discussed potential problems with the model, such as collinearity and correlation of the error terms.

In this post, I'll once again compare scikit-learn and statsmodels, and will explore how to include interaction terms and non-linear relationships in the model.  I'll also discuss nuances and potential problems of the resulting models, and possible areas for improvement using more sophisticated techniques.

### Boston Housing Dataset

I'll make use of the classic Boston Housing Dataset for this working example.  This dataset, originally published by Harrison, D. and Rubinfeld, D.L in 1978 has become one of the more common toy datasets for regression analysis.  The dataset contains information on 506 census tracts around the Boston, MA area, and is available via scikit-learn.

```python
from sklearn import datasets

# Load boston
boston = datasets.load_boston()
```

`datasets.load_boston()` returns a scikit-learn `bunch` object.  We can view information about the dataset via the `DESCR` attribute.

```python
>>> print(boston.DESCR)
.. _boston_dataset:
Boston house prices dataset
---------------------------
**Data Set Characteristics:**  
    :Number of Instances: 506 
    :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.
    :Attribute Information (in order):
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's
```

Instead of using the `return_X_y` parameter that returns two numpy arrays, I'll create a pandas DataFrame for ease of use below.

```python
import pandas as pd

X_df = pd.DataFrame(boston.data, columns=boston.feature_names)
y_df = pd.DataFrame(boston.target, columns=["MEDV"])
boston_df = pd.concat(objs=[X_df, y_df], axis=1)
```

### Qualitative Predictors & Dummy Encoding

Most of the predictors in the Boston housing dataset are quantitative.  The `CHAS` variable however, indicating if the census tract is bounded by the Charles River or not is qualitative and has been pre-encoded as 0 or 1.

I'll make use of the encoded `CHAS` variable in the models below.  However, for demonstration and to familiarize myself with the procedure, below I create a new categorical field `crime_label` and encode it via scikit-learn's `OneHotEncoder()` class. 

```python
from sklearn.preprocessing import OneHotEncoder

# Creating crime_label field
boston_df = \
    (boston_df
     .assign(
        crime_label=pd.cut(boston_df["CRIM"],
                           bins=3,
                           labels=["low_crime", "medium_crime", "high_crime"]))
    )

# Converting crime_label field to NumPy array
crime_labels_ndarray = boston_df["crime_label"].to_numpy().reshape(-1, 1)

# Defining encoder
encoder = OneHotEncoder()

# Fitting encoder on array, and transforming
crime_labels_encoded = encoder.fit_transform(crime_labels_ndarray)

# Converting encoded array to DataFrame
crime_labels_df = pd.DataFrame(data=crime_labels_encoded.toarray(),
                               columns=encoder.get_feature_names())

# Concatenating with boston_df
boston_df = pd.concat(objs=[boston_df, crime_labels_df], axis=1)
```

`boston_df` now contains three addition fields, indicating the encoded values of `crime_label`.

```python
>>> boston_df.head()
      CRIM    ZN  INDUS  ...  x0_high_crime  x0_low_crime  x0_medium_crime
0  0.00632  18.0   2.31  ...            0.0           1.0              0.0
1  0.02731   0.0   7.07  ...            0.0           1.0              0.0
2  0.02729   0.0   7.07  ...            0.0           1.0              0.0
3  0.03237   0.0   2.18  ...            0.0           1.0              0.0
4  0.06905   0.0   2.18  ...            0.0           1.0              0.0
```

Note, it is common in some use cases to drop an encoded column, as this column can be inferred explicitly from the other columns.  This can be accomplished by passing `drop="first` to `OneHotEncoder()`.  

```python
encoder = OneHotEncoder(drop="first")
```

It is also possible to achieve a similar result using pandas' `get_dummies()`.

```python
pd.get_dummies(boston_df["crime_label"], drop_first=True)
```

### Removing the Additive Assumption: Interaction Terms

Next, I'll explore relaxing the additive assumption of the multiple linear regression model.  In particular, I'll train a model on `MEDV` versus `ZN`, `CHAS` and `RM`.  For ease of use, user readability, and statistical inference results, I'll use the formula interface provided by statsmodels first, and then scikit-learn's interface further below.

Let's first train a model with no interaction terms, for comparison purposes.
```python
>>> import statsmodels.formula.api as smf
>>> model = smf.ols(formula="MEDV ~ ZN + CHAS + RM", data=boston_df)
>>> result = model.fit()
>>> result.summary()
<class 'statsmodels.iolib.summary.Summary'>
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   MEDV   R-squared:                       0.522
Model:                            OLS   Adj. R-squared:                  0.519
Method:                 Least Squares   F-statistic:                     182.5
Date:                Mon, 25 Jan 2021   Prob (F-statistic):           5.18e-80
Time:                        08:35:33   Log-Likelihood:                -1653.6
No. Observations:                 506   AIC:                             3315.
Df Residuals:                     502   BIC:                             3332.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept    -30.4697      2.654    -11.480      0.000     -35.684     -25.255
ZN             0.0666      0.013      5.182      0.000       0.041       0.092
CHAS           4.5212      1.126      4.017      0.000       2.310       6.733
RM             8.2635      0.428     19.313      0.000       7.423       9.104
==============================================================================
Omnibus:                      116.852   Durbin-Watson:                   0.757
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              675.350
Skew:                           0.866   Prob(JB):                    2.24e-147
Kurtosis:                       8.388   Cond. No.                         248.
==============================================================================
Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""
```

From the model summary, we see that all the variables are significant.  Let's take a look at the residual plot to see if any patterns standout.

```python
import seaborn as sns

y = boston_df["MEDV"]
y_hat = result.predict()
resid = y - y_hat

sns.scatterplot(x=y_hat, y=resid).set(xlabel="Predicted Value", 
                                      ylabel="Residual")
```

![2021-01-20-multiple-linear-regression-005-fig-1.png](/assets/img/2021-01-20-multiple-linear-regression-005-fig-1.png){: .mx-auto.d-block :}

This residual plot does appear to have a pattern.  Response values in the middle of the plot tend to be overestimates (negative residuals), while response values on the left and right of the plot tend to be underestimates (positive residuals).  

I'll explore adding some interaction terms here to the model to see if the fit is improved.  In the formula notation, a colon `:` is used to indicate an interaction term.  I could have also used an asterisk `*` to indicate an interaction term as well as the main effects.

```python
>>> model = smf.ols(formula="MEDV ~ ZN + CHAS + RM + ZN:CHAS + CHAS:RM + ZN:RM", data=boston_df)
>>> result = model.fit()
>>> result.summary()
<class 'statsmodels.iolib.summary.Summary'>
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   MEDV   R-squared:                       0.537
Model:                            OLS   Adj. R-squared:                  0.531
Method:                 Least Squares   F-statistic:                     96.35
Date:                Mon, 25 Jan 2021   Prob (F-statistic):           3.88e-80
Time:                        19:02:17   Log-Likelihood:                -1645.6
No. Observations:                 506   AIC:                             3305.
Df Residuals:                     499   BIC:                             3335.
Df Model:                           6                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept    -26.4011      2.980     -8.860      0.000     -32.256     -20.547
ZN            -0.4591      0.132     -3.467      0.001      -0.719      -0.199
CHAS           5.0981      9.048      0.563      0.573     -12.679      22.875
RM             7.6154      0.480     15.870      0.000       6.673       8.558
ZN:CHAS       -0.0320      0.065     -0.490      0.625      -0.161       0.097
CHAS:RM       -0.0813      1.411     -0.058      0.954      -2.853       2.690
ZN:RM          0.0783      0.020      3.978      0.000       0.040       0.117
==============================================================================
Omnibus:                      116.146   Durbin-Watson:                   0.715
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              651.824
Skew:                           0.869   Prob(JB):                    2.87e-142
Kurtosis:                       8.282   Cond. No.                     5.88e+03
==============================================================================
Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 5.88e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
"""
```

The above model fit is slightly improved with a $R^2$ statistic of 0.537, compared to 0.522 for the model with no interaction terms.  The pattern in the residual plot (not shown) is also slightly improved, but these improvements are minor.  

Worth note as well is that the terms `CHAS`, `ZN:CHAS` and `CHAS:RM` are all no longer significant.  Thus, I'll drop these terms from all subsequent models.

Instead of using the formula interface, I could have also fit these models using the `endog` and `exog` parameters of the statsmodels `OLS()` function.  Below is an example, fitting the same model as above, but removing the terms that are no longer significant.

```python
import statsmodels.api as sm

X = boston_df[["ZN", "RM"]].assign(ZN_RM=boston_df["ZN"] * boston_df["RM"])
X = sm.add_constant(X)
y = boston_df["MEDV"]

model = sm.OLS(endog=y, exog=X)
result = model.fit()
result.summary()
```

In below example, I'll show how polynomial regression models can be fit via the scikit-learn API.

### Removing the Linear Assumption: Polynomial Regression

In addition to relaxing the additive assumption of the linear regression model, next I'll explore relaxing the linear assumption by including some polynomial terms.

Using the `PolynomialFeatures()` class from scikit-learn's `preprocessing` module, I'll train a polynomial regression model.  Note, the `degree` parameter of `PolynomialFeatures()` has a default value of 2 indicating each predictor and each interaction term up to the power of 2 will be included in the model,

```python
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures()
X = poly.fit_transform(X=boston_df[["ZN", "RM"]])
y = boston_df["MEDV"]

model = linear_model.LinearRegression()
model.fit(X=X, y=y)

model.score(X=X, y=y)
```

The resulting model has an $R^2$ statistic of 0.582.  While this is a sizable improvement over the above value of 0.537, it indicates a large amount of the variation of the response variable is still left unexplained.  Undoubtedly, including more predictors in the model and using more sophisticated techniques would result in a better fit.

### Plotting in 3D

Just for fun - and because everyone loves a three-dimensional scatterplot - we can also view our regression models in 3D using plotly.  Below, I create a 3D scatterplot for a simpler model with only a single interaction term and no polynomial terms.

```python
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Define mash size for prediction surface
mesh_size = .02

# Train model
model = smf.ols(formula="MEDV ~ ZN + RM + ZN:RM", data=boston_df)
result = model.fit()

# Define x and y ranges
# Note: y here refers to y-dimension, not response variable
x_min, x_max = boston_df["ZN"].min(), boston_df["ZN"].max()
y_min, y_max = boston_df["RM"].min(), boston_df["RM"].max()
xrange = np.arange(x_min, x_max, mesh_size)
yrange = np.arange(y_min, y_max, mesh_size)
xx, yy = np.meshgrid(xrange, yrange)

# Get predictions for all values of x and y ranges
pred = model.predict(params=result.params, exog=np.c_[np.ones(shape=xx.ravel().shape), xx.ravel(), yy.ravel(), xx.ravel()*yy.ravel()])

# Reshape predictions to match mesh shape
pred = pred.reshape(xx.shape)

# Plotting
fig = px.scatter_3d(boston_df, x='ZN', y='RM', z='MEDV')
fig.update_traces(marker=dict(size=5))
fig.add_traces(go.Surface(x=xrange, y=yrange, z=pred, name='predicted_MEDV'))
fig.show()
```

![2021-01-20-multiple-linear-regression-005-fig-2.png](/assets/img/2021-01-20-multiple-linear-regression-005-fig-2.png){: .mx-auto.d-block :}


### Discuss potential problem stuff below


Discuss these 6 problems (maybe not all, but some):
1. Non-linearity of the response-predictor relationships.  <<-- residual plot
2. Correlation of error terms.
3. Non-constant variance of error terms.
4. Outliers.                <<-- maybe studentized residuals
5. High-leverage points.    <<-- leverage statistics
6. Collinearity.            <<-- VIF