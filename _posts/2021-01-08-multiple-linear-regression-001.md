---
layout: post
title: Multiple Linear Regression in Python, \#1 
subtitle: An Exploration and Comparison of scikit-learn vs. statsmodels
comments: false
---

Multiple linear regression is an extension of the simple linear regression model.  In its simplest form it is a relatively inflexible method that can produce insightful results and accurate predictions under certain conditions.  However, multiple linear regression has also been shown to be highly extensible, and many of these extensions (such as ridge regression, lasso regression, and logistic regression) vastly expand the usefulness and applicability of the linear regression paradigm.

In this post, I'll briefly introduce the multiple linear regression model.  I'll discuss fitting of the model, estimating coefficients, and assessing model accuracy.  Lastly, I'll also compare and contrast the Python packages scikit-learn and statsmodels as they relate to statistical inference and prediction using the multiple linear regression model.  For this brief introduction, I'll constrain myself to only considering quantitative predictors.  In future posts, qualitative predictors as well as interaction terms will be explored.

## Brief Introduction to Multiple Linear Regression

In contrast to simple linear regression, multiple linear regression is able to handle multiple predictor variables, which is a much more common situation in practice.  In general, the multiple linear regression model takes the form 

**NOTE: Make sure LaTeX is correct**
$$
\begin{aligned} Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_pX_p + \varepsilon \end{aligned}
$$

where $X_j$ represents the $j$th predictor and $\beta_j$ quantifies the association between that variable and the response.  $\beta_j$ is interpreted as the average effect on $Y$ of a one unit increase in $X_j$ *holding all other predictors fixed*.

### Estimating Coefficients

The multiple linear regression model is typically fit via the least squares method.  This method approximates the values $\beta_0$, $\beta_1$,..., $\beta_p$ by determining the values $\hat{\beta_0}$, $\hat{\beta_1}$,..., $\hat{\beta_p}$ that minimize the sum of squared residuals 

$$
\begin{aligned} \sum_{i=1}^{n} ((y_i) - \hat{y_i})^2 \end{aligned}.
$$

For simple linear regression in a two-dimensional space, this fitting method results in a line passing through the data.  However, for multiple linear regression, the method of least squares fitting results in a hyperplane that minimizes the squared distance between each point and the closest point on the plane.

**NOTE: Insert 3D graph from book here (provide the exact source -> p. 73, title, authers or maybe James et al.)**

After finding our estimated coefficients $\hat{\beta_0}$, $\hat{\beta_1}$,..., $\hat{\beta_p}$, it is natural to wonder how accurately each value estimates the true values $\beta_0$, $\beta_1$,..., $\beta_p$.  In general, the *standard error* of the estimate can be calculated to answer this question.  To continue with the comparison to simple linear regression, the intercept estimate $\hat{\beta_0}$ and the coefficient estimate $\hat{\beta_1}$ under this model can be computed via the following formulas:

$$
\begin{aligned} SE(\hat{beta_0})^2 = \sigma^2[1/n + ] \end{aligned}.
$$










# Notes for blog post #1 below ========
* introduction to multivariable linear regression (in practice, often more than one predictor)
* intro the data sets and where I got it from
* introduce what I'll cover (fitting the model, estimating coefficients, assessing model accuracy, comparison of scikit-learn and statsmodels)

* this post will focus on only quantitative predictors
* in a future post, will explore qualitative predictors and interactions

* notes on multivariable linear regression
    - extension of the simple linear regression model (estimating regression coefficienets -> 3D model of plane)
    - minimizing the squared distance
* note on estimating coefficients: discuss getting SE for slope term and confidence interval and p-values & hypothesis test
* note on assessing model accuracy: 
    - "extent to which the model fits the data"
    - R-squared & RSS 
    
* explore the 4 questions that start on p.75

* dive into code and nuances of each package






$$
\begin{aligned}
  & \phi(x,y) = \phi \left(\sum_{i=1}^n x_ie_i, \sum_{j=1}^n y_je_j \right)
  = \sum_{i=1}^n \sum_{j=1}^n x_i y_j \phi(e_i, e_j) = \\
  & (x_1, \ldots, x_n) \left( \begin{array}{ccc}
      \phi(e_1, e_1) & \cdots & \phi(e_1, e_n) \\
      \vdots & \ddots & \vdots \\
      \phi(e_n, e_1) & \cdots & \phi(e_n, e_n)
    \end{array} \right)
  \left( \begin{array}{c}
      y_1 \\
      \vdots \\
      y_n
    \end{array} \right)
\end{aligned}
$$

* section on filtering the data and preparing it


