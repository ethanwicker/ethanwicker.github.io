---
layout: post
title: "Exploring a pandas to scikit-learn Workflow"
subtitle: "Using scikit-learn's `ColumnTransformer` and `Pipeline` for encoding, imputing and scaling features"
comments: false
---

I recently read through this [excellent Medium article](https://medium.com/dunder-data/from-pandas-to-scikit-learn-a-new-exciting-workflow-e88e2271ef62) about the `ColumnTransformer` class in scikit-learn and how it can be used in tandem with `Pipeline`s and the `OneHotEncoder` class.  To strengthen my own understanding of the concept, I decided to follow the post with my own working example, and summarize the concepts along the way.

### Introduction

In scikit-learn's 0.20 release, the `ColumnTransformer` estimator was released.  This estimator allows different transformers to be applied to different fields of the data in parallel, before concatenating them together.  In particular, this estimator is attractive when processing a pandas DataFrame with categorical fields.

For this working example, I'll be using the same slimmed down Titanic dataset from my previous [logistic regression post](https://ethanwicker.com/2021-01-27-logistic-regression-002/).

Below, I'll make use of the `ColumnTransformer` estimator to encode two categorical fields via scikit-learn's `OneHotEncoder`.  Because scikit-learn machine learning models require their input to be two-dimensional numerical arrays, an encoding preprocessing step is required.

I'll also use the `SimpleImputer` and `StandardScaler` estimators to preform some preprocessing of numerical fields.  Lastly, I'll perform a regularized logistic regression and wrap all of these steps into a reusable and convenient `Pipeline`.

For example purposes further along in this post, I'll take the last 10 rows of `titanic` as a test dataset, and allocate the remaining rows as my training data.

```python
titanic_train = titanic.iloc[:-10]
titanic_test = titanic.iloc[-10:]
```

### OneHotEncoder

Let's first encode `sex` to demonstrate some functionality of `OneHotEncoder`.

```python
from sklearn.preprocessing import OneHotEncoder

# Initializing one-hot encoder
# Forcing dense matrix to be returned
encoder = OneHotEncoder(sparse=False)

# Encoding categorical field
sex_train = titanic_train[["sex"]]
sex_train_encoded = encoder.fit_transform(sex_train)

>>> sex_train_encoded
array([[0., 1.],
       [1., 0.],
       [1., 0.],
       ...,
       [0., 1.],
       [0., 1.],
       [1., 0.]])
```

From the output, we can see that the `male` and `female` values of `sex` have been encoded into two binary columns.

Notice that a NumPy array was returned.  We can access column names indicating which feature of `sex` is represented by each column using the `get_feature_names()` method.

```python
>>> feature_names = encoder.get_feature_names()
>>> feature_names
array(['x0_female', 'x0_male'], dtype=object)
```

We can also use the `inverse_transform()` method to return the original categorical label from the `sex` column.  Notice the brackets around `sex_train_encoded[0]` that force a list to be returned instead of a NumPy array.

```python
# Inverse transforming the first row
>>> encoder.inverse_transform([sex_train_encoded[0]])
array([['male']], dtype=object)

# Inverse transforming all rows
>>> encoder.inverse_transform(sex_train_encoded)
array([['male'],
       ['female'],
       ['female'],
       ...,
       ['male'],
       ['male'],
       ['female']], dtype=object)

# Verifying arrays are equivalent after inverse transforming
>>> np.array_equal(encoder.inverse_transform(sex_train_encoded), 
                   sex_train)
True
```

### Applying Transformations to Training & Test Sets

Whenever we transform a training field, we must also transform the corresponding field in the test set.  We must do this *after* splitting up the training and test datasets, instead of performing the transformation first and then splitting the data.  If we used the latter method here, some information from our test set would leak over into our training set.  This mistake is sometimes referred to as *data leakage*.

```python
# Encoding the sex column of our test dataset
>>> sex_test = titanic_test[["sex"]].copy()
>>> encoder.transform(sex_test)
array([[1., 0.],
       [0., 1.],
       [1., 0.],
       [0., 1.],
       [0., 1.],
       [1., 0.],
       [0., 1.],
       [1., 0.],
       [0., 1.],
       [0., 1.]])
```

The above transformation works great.  Because we have already initialized our one-hot encoder, we do not need to do so again.  

Although the above example worked well, occasionally we'll run into problems when transforming our test set.

#### Problem #1: ValueError: Found unknown categories

Occasionally, a value will be present in our test set that is not present in our training set.  This can present a problem, as our initialized one-hot encoder is expecting the same unique label values as the training set.

Let's suppose the first value of `sex_test` was misspelled `fmale` instead of `female`.

```python
>>> sex_test.iloc[0, 0] = "fmale"

>>> sex_test.head()
        sex
704   fmale
705    male
706  female
707    male
708    male
```

If we attempt the transformation on this new `DataFrame`, we'll get an error indicating an unknown category was found.

```python
>>> encoder.transform(sex_test)
ValueError: Found unknown categories ['fmale'] in column 0 during transform
```

In practice, we should investigate this issue further.  However, for this example, let's initialize a new `OneHotEncoder` with the `handle_unknown` parameter set to `"ignore"`, and fit it on the training observations again. Then, when we attempt to transform the test observations, the unknown values will be encoded as a row of all 0's.

```python
# Initializing new encoder
>>> encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

# Fitting on training observations
>>> encoder.fit(sex_train)

# Transforming test observations
# Notice first row is all 0's
>>> encoder.transform(sex_test)
array([[0., 0.],
       [0., 1.],
       [1., 0.],
       [0., 1.],
       [0., 1.],
       [1., 0.],
       [0., 1.],
       [1., 0.],
       [0., 1.],
       [0., 1.]])
```

#### Problem #2: Missing Values

Handling missing values in our test set is similar to handling unknown values in our test set.  If we initialize our encoder with `handle_unknown="ignore"`, these missing observations will be gracefully handled and encoded as rows of all 0's.

Note, it appears the 0.23.2 release of scikit-learn is able to have `None` values but not `NaN` values.


```python
# Assigning None to some elements of sex_test
>>> sex_test.iloc[1, 0] = None
>>> sex_test.iloc[2, 0] = None
>>> sex_test.head()

# Encoding
# Notice first three rows are all 0's
>>> encoder.transform(sex_test)
array([[0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 1.],
       [0., 1.],
       [1., 0.],
       [0., 1.],
       [1., 0.],
       [0., 1.],
       [0., 1.]])
```

#### Imputing Missing Values

In the event where we do have missing data, it can be useful to *impute* the missing values.  We can use scikit-learn's `SimpleImputer` transformer for this from the `impute` module.

Let's artificially create an `NaN` value in `sex_train_copy` below, and impute this missing values.  We can use the `strategy` parameter to control how the imputation is done.  For numerical data, `strategy` can be set to either `mean` or `median`.   For categorical data, we can set `strategy` to `constant`, which will allow us to set a constant string value to convert missing values to.  We can also set `strategy="most_frequent"` for both numerical and categorical observations, which will replace missing values with the most frequent observation in that column.

When `strategy` is equal to `"constant"`, we can optionally use the `fill_value` parameter to create the constant string value.  Below we'll just use the default value of `missing_value`.

```python
from sklearn.impute import SimpleImputer

# Assigning first element as NaN
sex_train_copy = sex_train.copy()
sex_train.iloc[0, 0] = np.nan

# Initializing SimpleImputer
# fill_value="missing_value" by default
simple_imputer = SimpleImputer(strategy="constant")

# Fitting and transforming
sex_train_copy_imputed = simple_imputer.fit_transform(sex_train_copy)
sex_train_copy_imputed
```

Now, we can use the `fit_transform()` method as before for encoding.

```python
>>> encoder.fit_transform(sex_train_copy_imputed)
array([[0., 0., 1.],
       [1., 0., 0.],
       [1., 0., 0.],
       ...,
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.]])
```

### scikit-learn `Pipeline`s

Instead of manually applying multiple fitting and transformation steps, we can instead use a `Pipeline`.  A `Pipeline` allows a list of transformations to be successively run, and a model can also be trained as the last estimator.  `Pipeline`s are especially useful for reproducibe workflows, such as applying the same transformation to training and test sets or different subsets of a dataset during cross validation.

Each step in the `Pipeline` consists of a two-item tuple.  The first element of the tuple is a string that labels the step, and the second element is an initialized estimator.  The output of each previous step will be the input to the next step.

```python
from sklearn.pipeline import Pipeline

# Creating steps
step_simple_imputer = ("simple_imputer", SimpleImputer(strategy="constant"))
step_encoder = ("encoder", OneHotEncoder(sparse=False, handle_unknown="ignore"))

# Creating pipeline
pipeline = Pipeline([step_simple_imputer, step_encoder])

# Fitting and transforming
pipeline.fit_transform(sex_train_copy)
```

After fitting the `Pipeline` on the training data, it is easy to transform the test data.  Notice that because the pipeline has already been fit, we do not need to refit it below, and can instead just call `transform()`. 

```python
pipeline.transform(sex_test)
```

#### Transforming Multiple Categorical Columns

It is simple as well to use our `Pipeline` on multiple categorical columns.  Just refit the `Pipeline` and run the transformation.

```python
multiple_fields = titanic_train[["sex", "ticket_class"]]
pipeline.fit_transform(multiple_fields)
```

#### Accessing Steps in our Pipeline

We can also access the individual steps of our `Pipeline` via the `named_steps` attribute.  For example, we can access the feature names via `get_feature_names()` method after specifying `encoder`.

```python
encoder = pipeline.named_steps["encoder"]
encoder.get_feature_names()
```

### scikit-learn's `ColumnTransformer`

The `ColumnTransformer` estimator from the `compose` module allows the user to control which columns get which transformation.  This is especially useful when we consider the different transformations categorical and numerical fields will need.

The `ColumnTransformer` takes a three-item tuple of the following structure:

```python
("name_of_column_transformer", "SomeTransformer(parameters), columns_to_transform")
```

The `columns_to_transform` could be a list of column names or integer indices, a boolean array, or a function that resolves to a selection of columns.

#### Passing a `Pipeline` to a `ColumnTransformer`

We can also pass a `Pipeline` to the `SomeTransformer(parameters)` input above.  Notice the below `Pipeline` is equivalent to the one described above, but `_categorical` has been appended throughout.  We'll have a numerical equivalent version shortly.

```python
from sklearn.compose import ColumnTransformer

# Creating steps
step_simple_imputer_categorical = ("simple_imputer", SimpleImputer(strategy="constant"))
step_encoder_categorical = ("encoder", OneHotEncoder(sparse=False, handle_unknown="ignore"))

# Creating pipeline
pipeline_categorical = Pipeline([step_simple_imputer_categorical, step_encoder_categorical])

# Defining categorical columns
columns_categorical = ["sex", "ticket_class"]

# Defining column transformer
transformers_categorical = [("transformers_categorical",
                             pipeline_categorical,
                             columns_categorical)]

# Creating column transformer
column_transformer = ColumnTransformer(transformers=transformers_categorical)
```

Because our `ColumnTransformer` selects columns, we can pass the entire `titanic_train` `DataFrame` to it.  The defined columns will be select and transformed as appropriate.  We could of course pass `titanic_test` to our `ColumnTransformer` as well.

```python
>>> column_transformer.fit_transform(titanic_train)
array([[0., 1., 0., 0., 1.],
       [1., 0., 1., 0., 0.],
       [1., 0., 0., 0., 1.],
       ...,
       [0., 1., 0., 0., 1.],
       [0., 1., 0., 0., 1.],
       [1., 0., 1., 0., 0.]])
```

To get the feature names of our encoded variables as we have done before, we need to use the `named_transformers_` attribute of our `ColumnTransformer`.  After doing so, we can then use the `named_steps` attribute of our pipeline as before.

```python
(column_transformer
    .named_transformers_["transformers_categorical"]
    .named_steps["encoder"]
    .get_feature_names())
```

#### Transforming Numeric Columns

Next, let's transform our numeric columns.  We'll impute missing numeric values using the median of that column, and then standardize the values.

```python
from sklearn.preprocessing import StandardScaler

# Creating steps
step_simple_imputer_numeric = ("simple_imputer", SimpleImputer(strategy="median"))
step_standard_scaler_numeric = ("standard_scaler", StandardScaler())

# Creating pipeline
pipeline_numeric = Pipeline([step_simple_imputer_numeric,
                             step_standard_scaler_numeric])

# Defining numeric columns
columns_numeric = ["age", "fare"]

# Defining column transformer
transformers_numeric = [("transformers_numeric",
                         pipeline_numeric,
                         columns_numeric)]

# Creating column transformer
column_transformer = ColumnTransformer(transformers=transformers_numeric)
```

Just as before, we can fit the `ColumnTransformer` directly to our `DataFrame`, and then transform it as appropriately.

```python
>>> column_transformer.fit_transform(titanic_train)
array([[-0.52929637, -0.52052854],
       [ 0.56642281,  0.68305589],
       [-0.25536658, -0.50784109],
       ...,
       [-0.66626127, -0.47173729],
       [-0.73474371, -0.50838994],
       [ 1.79910688,  0.90626108]])
```

### Combining Categorical and Numeric Column Transformers

Next, let's modify our `ColumnTransformer` structure so that we can perform both the categorical and numeric transformations in parallel.  The two resulting transformed NumPy arrays will be concatenated together into one array. 

```python
# Defining column transformer
>>> transformers = \
        [("transformers_categorical", pipeline_categorical, columns_categorical),
        ("transformers_numeric", pipeline_numeric, columns_numeric)]

# Creating column transformer
>>> column_transformer = ColumnTransformer(transformers=transformers)

# Transforming both categorical and numeric columns
>>> column_transformer.fit_transform(titanic_train)
array([[ 0.        ,  1.        ,  0.        , ...,  1.        , -0.52929637        ,-0.52052854],
       [ 1.        ,  0.        ,  1.        , ...,  0.        ,  0.56642281        , 0.68305589],
       [ 1.        ,  0.        ,  0.        , ...,  1.        , -0.25536658        ,-0.50784109],
       ...,
       [ 0.        ,  1.        ,  0.        , ...,  1.        , -0.66626127        ,-0.47173729],
       [ 0.        ,  1.        ,  0.        , ...,  1.        , -0.73474371        ,-0.50838994],
       [ 1.        ,  0.        ,  1.        , ...,  0.        ,  1.79910688        ,0.90626108]])
```

### Training a Machine Learning Model

Next, let's update our `Pipeline` to feed our transformed data into a machine learning model.  We'll train a logistic regression model below.  Unlike in my prior posts, this time I will make use of `LogisticRegression()`'s default regularization since we standardized our numeric predictor variables.

Below, we'll just use the `fit()` method instead of `fit_transform()`, because our final step in the `Pipeline` will be to actually fit the model.

```python
from sklearn.linear_model import LogisticRegression

# Creating steps
step_column_transformers = ("column_transformers", column_transformer)
step_logistic_regression = ("logistic_regression", LogisticRegression())

# Creating pipeline
log_reg_pipeline = Pipeline([step_column_transformers, step_logistic_regression])

# Assigning y
y = titanic_train["survived"]

# Transforming data and fitting model
log_reg_pipeline.fit(titanic_train, y)
```

We can use the `score()` method to return the correct classification rate.

```python
>>> log_reg_pipeline.score(titanic_train, y)
0.7911931818181818
```

### Cross-validation

Of course, the above correct classification rate value indicates the results on the training set.  To get a better idea of how our model might perform on test data, let's perform a 10-fold cross-validation.

```python
>>> from sklearn.model_selection import KFold, cross_val_score

# Initializing k-fold cross-validator
>>> k_fold = KFold(n_splits=10, shuffle=True, random_state=123)

# Getting cross-validation scores
>>> cross_val_scores = cross_val_score(estimator=log_reg_pipeline, 
                                       X=titanic_train, 
                                       y=y, 
                                       cv=k_fold)

# Getting average cross-validation score
>>> cross_val_scores.mean()
0.785513078470825
```

### Grid Search

Lastly, let's perform a grid search to determine the optimal values of our transformation and fitting procedures.  We'll pass a dictionary object to scikit-learn's `GridSearchCV`.  We'll need to put double underscores between the name of each layer in our `Pipeline`, as well as the actual parameter name.

```python
from sklearn.model_selection import GridSearchCV

# Defining parameter grid
param_grid = {
    "column_transformers__transformers_numeric__simple_imputer__strategy":
        ["mean", "median"],
    "logistic_regression__C":
        [.0001, .001, .01, .1, 1, 10, 100, 1000]
}

# Initializing grid search
grid_search = GridSearchCV(estimator=log_reg_pipeline,
                           param_grid=param_grid,
                           cv=k_fold)

# Fitting grid search
grid_search.fit(titanic_train, y)
```

We can also view the best parameter combination and the best score.

```python
>>> grid_search.best_params_
{'column_transformers__transformers_numeric__simple_imputer__strategy': 'mean', 'logistic_regression__C': 10}

>>> grid_search.best_score_
0.7869215291750503
```

We can view detailed results as a pandas `DataFrame` as well.

```python
>>> pd.DataFrame(grid_search.cv_results_)
    mean_fit_time  std_fit_time  ...  std_test_score  rank_test_score
0        0.201999      0.415343  ...        0.066751               15
1        0.014747      0.000764  ...        0.060855               13
2        0.016023      0.001118  ...        0.038972               11
3        0.019522      0.003494  ...        0.055859                7
4        0.020603      0.002647  ...        0.059064                9
5        0.018984      0.003322  ...        0.060287                1
6        0.016421      0.000717  ...        0.060287                1
7        0.016278      0.000662  ...        0.060287                1
8        0.015751      0.000653  ...        0.066751               15
9        0.016661      0.002365  ...        0.060855               13
10       0.017589      0.001195  ...        0.038972               11
11       0.018600      0.001226  ...        0.055859                7
12       0.018348      0.000565  ...        0.059064                9
13       0.018073      0.001442  ...        0.060287                1
14       0.016373      0.000261  ...        0.060287                1
15       0.017112      0.001278  ...        0.060287                1
[16 rows x 20 columns]
```
