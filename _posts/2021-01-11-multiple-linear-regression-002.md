---
layout: post
title: "Multiple Linear Regression #2"
subtitle: A Comparison of Python's scikit-learn and statsmodels Libraries
comments: false
---

## ADD MY IMPORT STATEMENTS AT THE TOP OF EACH PIECE TO MAKE IT CLEAR
## maybe link to a github repo with organized code? might not do right now and may just focus on the blog

NOTE: Make sure to spell precipitation right.  This word is definitely spelled wrong.

Note: Add headers where appropriate.

Should add titles to my plots, and show the code for how they were made.

Idea for future post to expand on this.  
Look at per hour weather averages, but also group on carrier and create a "morning", "midday", "evening", "night" variable to predict this as well.  Maybe weather at night is worse, or maybe flights get more detailed as the day goes on.  

```python
import numpy as np NO <-- do I need?
import pandas as pd NO
import seaborn as sns NO
from nycflights13 import airlines, flights, weather NO
from sklearn.linear_model import LinearRegression NO
import statsmodels.api as sm N
import statsmodels.formula.api as smf N
```

This post is the second in a series on the multiple linear regression model.  In a [previous post](https://ethanwicker.com/2021-01-08-multiple-linear-regression-001/), I introduced the model and much of it's associated theory.  In this post, I'll continue exploring the multiple linear regression model with an example in Python, including a comparison of the scikit-learn and statsmodels libraries.

### Introducing the Example

As a working example, I'll explore the effect weather had on airline flights leaving three New York City airports in 2013.  In particular, I'll join the `airlines`, `flights`, and `weather` datasets from the [nycflights13](https://pypi.org/project/nycflights13/) Python package and investigate the relationship wind speed, precipitation and visibility has on departure delay.

### Preparing the Data

Fortunately, the datasets I'll be working with do not require much cleaning and organization before applying the model.  After renaming the fields for clarity and converting the `timestamp_hour` fields to a `datetime64` data type, I'll join the dataframes via pandas' `merge` and name it `nyc`.  I'll also drop missing values using `pd.dropna()`.  In practice, a more thorough investigation into these missing values would be needed, but I'll ignore this for demonstration purposes. 

```python
import pandas as pd
from nycflights13 import airlines, flights, weather


# Organizing airlines ---------------------------------------------------------

# Renaming fields for clarity
airlines = airlines.rename(columns={'carrier':'carrier_abbreviation',
                                    'name': 'carrier_name'})


# Organizing flights ----------------------------------------------------------

# Renaming fields for clarity
flights = flights.rename(columns={'dep_delay': 'departure_delay',
                                  'carrier': 'carrier_abbreviation',
                                  'origin': 'orig_airport',
                                  'time_hour': 'timestamp_hour'})

# Assigning timestamp_hour field to datetime type
flights = flights.assign(timestamp_hour=pd.to_datetime(flights.timestamp_hour, utc=True))


# Organizing weather ----------------------------------------------------------

# Renaming fields for clarity
weather = weather.rename(columns={'origin': 'orig_airport',
                                  'wind_speed': 'wind_speed_mph',
                                  'precip': 'precipitation_inches',
                                  'visib': 'visibility_miles',
                                  'time_hour': 'timestamp_hour'})

# Assigning timestamp_hour field to datetime type
weather = weather.assign(timestamp_hour=pd.to_datetime(weather.timestamp_hour, utc=True))


# Joining Data ----------------------------------------------------------------

# Joining
nyc = \
    flights \
    .merge(right=airlines, how='left', on='carrier_abbreviation') \
    .merge(right=weather, how='left', on=['orig_airport', 'timestamp_hour'])

# Selecting relevant fields and reordering
nyc = nyc[['timestamp_hour',
           'orig_airport',
           'carrier_name',           
           'departure_delay',           
           'wind_speed_mph',
           'precipitation_inches',
           'visibility_miles']]

# Dropping missing values
nyc = nyc.dropna()
```

At this point, the  `nyc` dataframe has been created.  A glimpse of the data structure can be seen below.

NOTE: How to make the output show here?
Maybe just copy and paste from the console
```python
nyc.head()

nyc.info()
```

Before diving into modeling, it's good practice to perform some initial exploratory data analysis.  I'll use the seaborn data visualization library to create some plots.

Because the data is so large, I'll take a small sample of 10,000 observations and create a pair plot.

```python
import seaborn as sns

nyc_small = nyc.sample(n=10000)

sns.pairplot(nyc_small)
```

From the pair plot, it's clear the data is quite noisy.  However, we can still learn a thing or two from the plot.  For example, most days in New York City tend to have good to mild weather with high visibility, low precipitation and moderate wind speeds.  Also, from this small sample, no striking patterns or correlation stand out.

Perhaps instead of attempting to understand how these weather patterns affect individual departure delays, we should aggregate our data per hour and explore how average weather patterns affect average departure delays.

```python
# Aggreaging data per hour
nyc_per_hour = \
    nyc\
    .groupby("timestamp_hour")\
    .agg(mean_departure_delay=('departure_delay', 'mean'),
         mean_wind_speed_mph=('wind_speed_mph', 'mean'),
         mean_precipitaton_inches=('precipitation_inches', 'mean'),
         mean_visibility_miles=('visibility_miles', 'mean'))\
    .reset_index()

### ALSO GIVE THIS PLOT LABELS AND A TITLE

sns.pairplot(nyc_per_hour)
```

The pair plot for the new aggregated `nyc_per_hour` dataset is still quite busy.  As a last plot, let's create a correlation heatmap to confirm the lack of correlation among the variables.

```python
import matplotlib.pyplot as plt
import numpy as np

labels =['Departure Delay', 'Wind Speed (MPH)', 'Precipitation (in)', 'Visibility (miles)']

sns\
    .heatmap(nyc_per_hour.corr(),
             mask=np.triu(np.ones_like(nyc_per_hour.corr())),
             annot=True,
             xticklabels=labels,
             yticklabels=labels)\
    .set_title('Correlation Heatmap of Average per Hour Values')

plt.xticks(rotation=45)
```

From the heatmap, we can see that the three predictor variables are mostly uncorrelated.  For our multiple linear regression model, this is preferred.  Multicollinearity among predictors can lead to spurious predictions and high variance among the regression coefficients.  A variety of techniques can be helpful when dealing with mutlicollinearity among predictors, such as lasso regression and principal component analysis.  These topics will be covered in more depth in later posts.

It is worth noting that there is a somewhat strong correlation between average precipitation and average visibility, with a Pearson correlation of -0.38.  This isn't surprising to see, as we might expect rainy days to have low visibility.  For the purposes of this demonstration, I'll ignore this multicollinearity.

From the heatmap as well, we can also see there is a small degree of correlation between average precipitation and average departure delay, with a Pearson correlation of 0.22, and a comparable negative correlation between average visibility and average departure delay with a Pearson correlation of -0.21. 


### Multiple Linear Regression Model via scikit-learn

After some exploratory analysis, we're ready for modeling.  I'll first train the model using scikit-learn, and then train the same model using statsmodels.  While scikit-learn is an excellent machine learning package, it isn't intended for statistical inference.  For this reason, I'll explore the model summary of statsmodels in-depth to learn more about the regression in the next section.

Fitting the model via the scikit-learn API is quite simple.  An *estimator* object is first created, and then is fit using the predictor variables `X` on the response variable `y`.

```python
from sklearn.linear_model import LinearRegression

# Assign estimator
linear_reg = LinearRegression()

# Defining predictors and response
X = nyc_per_hour[['mean_wind_speed_mph', 'mean_precipitaton_inches', 'mean_visibility_miles']]
y = nyc_per_hour['mean_departure_delay']

# Fitting model
linear_reg.fit(X=X, y=y)
linear_reg.coef_
linear_reg.intercept_
linear_reg.score(X, y) # R^2 value
```

After fitting the model, we can view the regression coefficients and intercept using the `coef_` and `intercept_` attributes.  We could also perform prediction with the fitted model using the `predict()` method.

Of particular interest to this post is the `score()` method.  For the linear regression model, this method returns $R^2$, or the coefficient of determination.  

```python
lin_model.score(X, y) # R^2 value
```



### Multiple Linear Regression Model via statsmodels





# Fitting with sm.OLS
X = per_hour[['mean_wind_speed_mph', 'mean_precipitaton_inches', 'mean_visibility_miles']]
X = sm.add_constant(X)


mod = sm.OLS(endog=per_hour.mean_departure_delay, exog=X)
result = mod.fit()
result.summary()


sns.pairplot(per_hour)


a=df.groupby('kind').agg(min_height=('height', 'min'),
                               max_weight=('weight', 'max'))

sns.lineplot(data=a, x="timestamp_hour", y="mean_departure_delay")

nyc.timestamp_hour.unique()


### Big header
#### Small header








$$
\begin{aligned} 
 
\end{aligned}
$$




* introduce the data and how I got it, with links to packages
* dive into code and nuances of each package
* section on filtering the data and preparing it
Python stuff - basically just all the code I wrote.
  
https://quicklatex.com/