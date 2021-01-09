---
layout: post
title: "Multiple Linear Regression #1, An Introduction"
subtitle: An Exploration and Comparison of scikit-learn vs. statsmodels
comments: false
---

Multiple linear regression is an extension of the simple linear regression model.  In its simplest form it is a relatively inflexible method that can produce insightful results and accurate predictions under certain conditions.  However, multiple linear regression has also been shown to be highly extensible, and many of these extensions (such as ridge regression, lasso regression, and logistic regression) vastly expand the usefulness and applicability of the linear regression paradigm.

In this post, I'll briefly introduce the multiple linear regression model.  I'll discuss fitting of the model, estimating coefficients, and assessing model accuracy.  Lastly, I'll also compare and contrast the Python packages scikit-learn and statsmodels as they relate to statistical inference and prediction using the multiple linear regression model.  For this brief introduction, I'll constrain myself to only considering quantitative predictors.  In future posts, qualitative predictors as well as interaction terms will be explored.

#### Multiple Linear Regression

In contrast to simple linear regression, multiple linear regression is able to handle multiple predictor variables, which is a much more common situation in practice.  In general, the multiple linear regression model takes the form 

$$
\begin{aligned} Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_pX_p + \epsilon \end{aligned}
$$

where $X_j$ represents the $j$th predictor and $\beta_j$ quantifies the association between that variable and the response.  $\beta_j$ is interpreted as the average effect on $Y$ of a one unit increase in $X_j$ *holding all other predictors fixed*.

#### Estimating Coefficients

The multiple linear regression model is typically fit via the least squares method.  This method approximates the values $\beta_0$, $\beta_1$,..., $\beta_p$ by determining the values $\hat{\beta_0}$, $\hat{\beta_1}$,..., $\hat{\beta_p}$ that minimize the sum of squared residuals 

$$
\begin{aligned} RSS = \sum_{i=1}^{n} (y_i - \hat{y_i})^2 \end{aligned}.
$$

For simple linear regression in a two-dimensional space, this fitting method results in a line passing through the data.  However, for multiple linear regression, the method of least squares fitting results in a hyperplane that minimizes the squared distance between each point and the closest point on the plane.  For the event where we have two predictor variables and one response variable, we can visualize the hyperplane as a two-dimensional plane in a three-dimensional space, but the multiple linear regression model is applicable to higher dimensional spaces as well.

| ![multiple-linear-regression-3d-plot-isl-ch3.png](/assets/img/multiple-linear-regression-3d-plot-isl-ch3.png){: .mx-auto.d-block :} |
| :--: |
| <sub><sup>**Source:** *Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani. An Introduction to Statistical Learning: with Applications in R. New York: Springer, 2013.* |

##### Standard Errors

After finding our estimated coefficients $\hat{\beta_0}$, $\hat{\beta_1}$,..., $\hat{\beta_p}$, it is natural to wonder how accurately each value estimates the true values $\beta_0$, $\beta_1$,..., $\beta_p$.  In general, the *standard error* of the estimate can be calculated to answer this question.  To continue with the comparison to simple linear regression, the intercept estimate $\hat{\beta_0}$ and the coefficient estimate $\hat{\beta_1}$ under this model can be computed via the following formulas:

$$
\begin{aligned} 
SE(\hat{\beta_0})^2 = \sigma^2\left[\frac{1}{n} + \frac{\bar{x}^2} {\sum_{i=1}^{n}(x_i-\bar{x})^2}\right], 
SE(\hat{\beta_1})^2 = \frac{\sigma^2}{\sum_{i=1}^{n}(x_i - \bar{x})^2}
\end{aligned}
$$

where $\sigma^2 = Var(\epsilon)$.  

Note, these standard error formulas assume that the errors $\epsilon_i$ for each observation are uncorrelated and have the same variance $\sigma^2$.  This assumption is rare in practice, but these standard error estimations still turn out to be a good approximation.  Similarly, in general, $\sigma^2$ is typically not known, but can be estimated from the data.  This estimation is known as the *residual standard error* and is given by $RSE = \sqrt{RSS/(n-2)}$.

##### Confidence Intervals and Hypothesis Testing

Returning to the multiple linear regression model, standard errors can also be used to computer confidence intervals and perform hypothesis tests.  For linear regression, a 95% confidence interval for $\beta_j$ is approximately equivalent to 

$$
\begin{aligned}
\hat{\beta_j} \pm 2 \cdot SE(\hat{\beta_j})
\end{aligned}.
$$

The most common hypothesis test involves testing the *null hypothesis* that there is no relationship between $X$ and $Y$.  Mathematically, this is equivalent to testing

$$
\begin{aligned}
H_0: \beta_j = 0
\end{aligned}
$$

versus 

$$
\begin{aligned}
H_a: \beta_j \neq 0
\end{aligned}.
$$

To test the null hypothesis, we need to quantify how far our estimated coefficient $\beta_j$ is from 0.  This can be determined by calculating a *t-statistic* given by

$$
\begin{aligned}
t = \frac {\hat{\beta_j}-0} {SE(\hat{\beta_j})}
\end{aligned}.
$$

This t-statistic is a measurement of the number of standard deviations that $\hat{\beta_j}$ is away from 0.  Via the Central Limit Theorem, we know that for large sample sizes, the t-distribution will approximate a Gaussian distribution, and thus we can calculate a *p-value* for $\hat{\beta_j}$.  For p-values below a predefined significance level - typically 0.05 or 0.01 - we *reject the null hypothesis*. 

#### Assessing Model Accuracy

The quality of a multiple linear regression fit is typically assessed via two values: the residual standard error (RSE) and the $R^2$ statistic.  These values are included as standard output in most statistical software, including Python's statsmodels module and the base R distribution.  In future post, I'll explore examples and discuss these values, but for now I'll constrain my discussion to only theory.

##### Residual Standard Error

The residual standard error was briefly introduced above in the context of calculating standard error values.  RSE, as defined above, is a measure of the standard deviation of the *irreducible error* term $\epsilon$.  If a model fit is quite accurate and predictions obtained from the model are very close to the true response values, RSE is quite small.  However, is a model fit is poor and predictions obtained from the model are far from the true response values, we can expect RSE to be quite high.  This, RSE is considered a measure of the *lack of fit* of the model.

Note, the RSE is measured in the units of $Y$, and what is considered *high* or *low* is dependent on the problem at hand.

##### $R^2$ Statistic

The $R^2$ statistic, commonly called the coefficient of determination or coefficient of multiple correlation, is an alternative measure of model fit.  In contrast to RSE which is measured in the units of $Y$, the $R^2$ statistic is a proportion.  The R^2$ statistic indicates the proportion of variance explained explain by the model and is independent of the scale of $Y$.  $R^2$ is calculated via the formula 

$$
\begin{aligned}
R^2 = \frac{TSS - RSS}{TSS} = 1 - \frac{RSS}{TSS}
\end{aligned}
$$

where $TSS = \sum_(y_i - \bar{y})^2 is the *total sum of squares* and is a measure of the total variance in the response $Y$.

The $R^2$ statistic measures the *proportion of variability in* $Y$ *that can be explained using* $X$.  A value close to 1 indicates that a large proportion of the variability in the response has been explained by the regression, while a value close to 0 indicates that the regression did not explain much of the variability in the response.  A low $R^2$ value can occur when a linear model is not a good approximation of the data, or the variance of $\epsilon$ is inherently high, or both.

An extension of the $R^2$ statistic known as Adjusted $R^2$ and denoted $R_adj^2$ will be discussed in a future post.

#### Determining if a Relationship Exists Between the Response and Predictors

A question we have yet to explore is that of whether there exists a relationship between the response and predictor variables.  It might be tempting to use the individual predictor p-values discussed above to make this assessment and claim that if any individual predictor p-values are small, then at least one of the predictors is related to the response.  However, this logic is flawed, especially when the number of predictors $p$ is large.  As the number of predictors increases, the chance that the p-value of a single predictor will appear significant increases, even if there is no true association between the predictors and the response.

Fortunately, the *F-statistic* does not suffer from this problem, as it adjusts for the number of predictors.

Mathematically, determining if a relationship exists between the response and predictor variables is equivalent to the null hypothesis

$$
\begin{aligned}
H_0: \beta_1 = \beta_2 = ... = \beta_p = 0
\end{aligned}
$$

versus the alternative

$$
\begin{aligned}
H_a: at least one \beta_j is non-zero.
\end{aligned}
$$

This hypothesis test is performed by calculating the F-statistic,

$$
\begin{aligned}
F = \frac{(TSS - RSS)/p}{RSS/(n-p-1)}
\end{aligned}
$$

If the assumptions of the linear model are correct, we can show that 

$$
\begin{aligned}
E{RSS/n-p-1} = \sigma^2
\end{aligned}
$$

and that, provided $H_0$ is true,

$$
\begin{aligned}
E{(TSS - RSS)/p} = \sigma^2,
\end{aligned}
$$

where $E{X}$ indicates the *expected value*.

When there is no relationship between the response and the predictors, we expect the F-statistic to be close to 1.  However, when $H_a$ is true, then $E{(TSS - RSS)/p} > \sigma^2$, and we expect F to be greater than 1.

To determine whether to reject the null hypothesis, we can calculate a p-value from the resulting F-statistic.  When $H_0$ is true and either the errors $\epsilon_i$ have a normal distribution, or the sample size $n$ is large, the F-statistic follows an F-distribution.  Dependent on the calculated p-value, we can either reject the null hypothesis and claim there is a relationship between the response and the predictors, or not reject the null hypothesis and make so such claim.

In the event where the F-statistic is low and we cannot claim that a relationship exists between the response and the predictors, we should not interpret any individual predictor p-values as significant, even if some p-values are low.

# Notes for blog post #1 below ========
$$
\begin{aligned}

\end{aligned}
$$

# Still to write
Update the intro with what I'm going to go through

 Assessing model accuracy -> RSE and R-squared
Some important questions:
1. Is there a relationship between response and predictors --> F-statistic, can be done on all or a subset of predictors
IMPORTANT: CANNOT JUST look at individual p-values, need to look at F-statistic.  After all, it seems likely that if any one of the p-values for the individual variables is very small, then at least one of the predictors is related to the response. However, this logic is flawed, especially when the number of predictors p is large.
   
2. Deciding on important variables:
variable selection methods -> mention what they are but discuss later
   
3. Model fit: same as assesing model accuracy, can just leave there 

4. predictions: how to make, confidence intervals vs prediction intervals

consider some headers above



* note on assessing model accuracy: 
    - "extent to which the model fits the data"
    - R-squared & RSS 
    
* explore the 4 questions that start on p.75

* introduce the data and how I got it, with links to packages
* dive into code and nuances of each package
* section on filtering the data and preparing it

