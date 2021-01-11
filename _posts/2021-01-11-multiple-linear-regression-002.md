---
layout: post
title: "Multiple Linear Regression #2"
subtitle: A Comparison of Python's scikit-learn and statsmodels Libraries
comments: false
---

## ADD MY IMPORT STATEMENTS AT THE TOP OF EACH PIECE TO MAKE IT CLEAR
## maybe link to a github repo with organized code? might not do right now and may just focus on the blog


```python
import numpy as np NO <-- do I need?
import pandas as pd NO
import seaborn as sns NO
from nycflights13 import airlines, flights, weather NO
from sklearn.linear_model import LinearRegression NO
import statsmodels.api as sm N
import statsmodels.formula.api as smf N
```

This post is the second in a series on the multiple linear regression model.  In a [previous post](https://ethanwicker.com/2021-01-08-multiple-linear-regression-001/), I introduced the model and much of it's associated theory.  In this post, I'll continue exploring the multiple linear regression model with an example in Python.  I'll also compare and contrast Python's scikit-learn and statsmodels libraries.

### Introducing the Example

As a working example, I'll explore the effect weather had on airline flights leaving three New York City airports in 2013.  In particular, I'll join the `airlines`, `flights`, and `weather` datasets from the [nycflights13](https://pypi.org/project/nycflights13/) Python package and investigate the relationship between wind speed, precipitation, visibility and departure delay.

### Preparing the Data

Since the nycflights package is intended to be used for example purposes, I fortunately do not have to do much data cleaning.

```python
import pandas as pd
from nycflights13 import airlines, flights, weather


# Organizing airlines ---------------------------------------------------------

# Renaming fields for clarity
airlines = airlines.rename(columns={'carrier':'carrier_abbreviation',
                                    'name': 'carrier_name'})


# Organizing flights ----------------------------------------------------------

# Renaming fields for clarity
flights = flights.rename(columns={'dep_time': 'departure_time',
                                  'sched_dep_time': 'scheduled_departure_time',
                                  'dep_delay': 'departure_delay',
                                  'arr_time': 'arrival_time',
                                  'sched_arr_time': 'scheduled_arrival_time',
                                  'arr_delay': 'arrival_delay',
                                  'carrier': 'carrier_abbreviation',
                                  'origin': 'orig_airport',
                                  'dest': 'dest_airport',
                                  'distance': 'distance_miles',
                                  'time_hour': 'timestamp_hour'})

# Assigning timestamp_hour field to datetime type
flights = flights.assign(timestamp_hour=pd.to_datetime(flights.timestamp_hour, utc=True))


# Organizing weather ----------------------------------------------------------

# Renaming fields for clarity
weather = weather.rename(columns={'origin': 'orig_airport',
                                  'wind_speed': 'wind_speed_mpg',
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
# NOTE: Likely don't need all of these fields, can probably remove some
# TAKE OUT SOME OF THESE FIELDS
nyc = nyc[['timestamp_hour',
           'orig_airport',
           'dest_airport',
           'carrier_name',
           'carrier_abbreviation',
           'scheduled_departure_time',
           'departure_time',
           'departure_delay',
           'scheduled_arrival_time',
           'arrival_time',
           'arrival_delay',
           'air_time',
           'distance_miles',
           'wind_speed_mpg',
           'precipitation_inches',
           'visibility_miles']]
```

Show what the data looks like here using
nyc.info()
How to actually get the output to display like in Rmarkdown?







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