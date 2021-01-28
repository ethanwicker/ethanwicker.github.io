---
layout: post
title: "Logistic Regression #1"
subtitle: A Brief Introduction, Maximum Likelihood Estimation, Multiclass Logistic Regression, and More
comments: false
---

This is the first post in a series on the logistic regression model.  The structure of this post was influenced by the fourth chapter of *An Introduction to Statistical Learning: with Applications in R* by Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani.

### Logistic Regression

In contrast to linear regression, which attempts to model the response $Y$ directly, logistic regression attempts to model the probability that $Y$ belongs to a particular category, or class.

Although we can use a linear regression model to represent probabilities $p(x)$, as in 

$$
\begin{aligned} 
p(x) = \beta_0 + \beta_1X_1,
\end{aligned}
$$

this approach is problematic.  In particular, for some values of $X_1$, this method will predict probability values below 0 and above 1.

To avoid this problem, we should instead model the probability that $Y$ belongs to a given class using a function that provides bounded output between 0 and 1.  Many functions could be used for this, including the *logistic function* used in logistic regression:

$$
\begin{aligned} 
p(x) = \frac{e^{\beta_0 + \beta_1X_1}}{1 + e^{\beta_0 + \beta_1X_1}}.
\end{aligned}
$$

This model is fit via the method of *maximum likelihood*, to be discussed below.  It will always produce an $S$ shaped function, bounded between 0 and 1.

We can further manipulate the function such that 

$$
\begin{aligned} 
\frac{p(x)}{1-p(x)} = e^{\beta_0 + \beta_1X_1}.
\end{aligned}
$$

The left-hand side of the above equation, $p(x) / (1 - p(x))$, is known as the *odds* and can take any value between 0 and $\infty$.  Odds values close to 0 indicate very low probabilities, and odds values close to $\infty$ indicate very high probabilities.

Finally, by taking the logarithm of both sides of the above equation we have

$$
\begin{aligned} 
log(\frac{p(x)}{1-p(x)}) = \beta_0 + \beta_1X_1.
\end{aligned}
$$

Here, the left-hand side is referred to as the *log-odds* or *logit*.  The logistic regression model has a logit that is linear with respect to $X$.  Thus, increasing $X$ by one unit changes the log odds by $B_1$, or equivalently, multiples the odds by $e^{B_1}$. 

#### Estimating the Regression Coefficients

In the multiple linear regression model, we used the method of least squares to estimate the linear regression coefficients.  In the logistic regression model, we could also use a (non-linear) least squares fitting procedure, but it is preferred to use the more general method of *maximum likelihood* due to its statistical properties.

Maximum likelihood estimation is a fitting method used to estimate regression coefficients.  The method seeks to estimate the coefficients $B_0, B_1, B_2, \cdots$ such that the predicted probability $\hat{p}(x)$ of the class assignment of each observation is as close as possible to the actual class assignment, for that observation.  Visually, the method of maximum likelihood estimation is akin to fitting the logistic function many times to the data with slight differences, and selecting the particular function that maximizes the correct probability for each class assignment.

Mathematically, this can be formalized via a *likelihood function*:

$$
\begin{aligned} 
\ell(\beta_0, \beta_1) = \prod_{i:y_i = 1}p(x_i) \prod_{i':y_{i'} = 0}(1 - p(x_i')),
\end{aligned}
$$

where $\beta_0$ and $\beta_1$ are chosen to maximize this likelihood function.

Maximum likelihood is a general approach used to fit many non-linear models.  In fact, in the linear regression setting, least squares is a special case of maximum likelihood.

When performing statistical inference, the summary of a logistic regression model can be interpreted in much the same way as that of a linear regression model.  In particular, coefficient p-values used to infer if the particular predictor is associated with the response are interpreted in the same manner.  It is worth noting, the estimated intercept values are of little interest; their main use is to adjust the average fitted probabilities to the proportion of those in the data.

#### Using Qualitative Predictors

Similar to the linear regression model, qualitative predictors can be included in the logistic regression model via the one-hot encoding (or dummy variable) method.

### Multiple Logistic Regression

The logistic regression model easily extends to multiple predictor variables as well.  We can generalize the logistic function as 

$$
\begin{aligned} 
p(X) = \frac{e^{\beta_0 + \beta_1X_1 + \cdots + \beta_pX_p}}{1 + e^{\beta_0 + \beta_1X_1 + \cdots + \beta_pX_p}}
\end{aligned}
$$

where $X = X_1, \cdots, X_p$ and $p$ is the number of predictors.  We can also rewrite this equation as

$$
\begin{aligned} 
log(\frac{p(X)}{1-p(X)}) = \beta_0 + \beta_1X_1 + \cdots + \beta_pX_p.
\end{aligned}
$$

Similar to the linear regression setting, it will sometimes be the case that results obtained using one predictor may be quite different from those obtained using multiple predictors, especially when there is correlation among the predictors.  In general, this phenomenon is known as *confounding*.

#### Multiclass Logistic Regression

Logistic regression is readily extendable to the case where there are greater than two response classes.  This is known as *multiclass logistic regression*, or sometimes *multinomial logistic regression*.  One such extension has the form

$$
\begin{aligned} 
Pr(Y = k|X) = \frac{e^{\beta_{0k} + \beta_{1k}X_1 + \cdots + \beta_{pk}X_p}}{\sum_{l=1}{K}e^{\beta_{0k} + \beta_{1k}X_1 + \cdots + \beta_{pk}X_p}}
\end{aligned}
$$

where $k$ is a class index out of $K$ possible classes.  In this form, there is a separate linear function for each class.  As in the binary logistic regression case, only $K-1$ linear functions are needed.

However, in practice, multiclass logistic regression is commonly used.  Instead, *discriminate analysis* (the topic of my next post) is often used instead.
