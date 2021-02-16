---
layout: post
title: "Cross-Validation #1"
subtitle: Validation Sets, Leave-One-Out Cross-Validation, and k-Fold Cross-Validation
comments: false
---

*Resampling methods* are a crucial tool used commonly in modern statistics and data science.  These methods involve taking repeated samples from a training dataset and refitting a model of interest on each individual sample to obtain additional information about the fitted model.  These methods allow us to learn new information about the model that might not be available from a single fit of the model.

Two of the most common resampling methods are *cross-validation* and the *bootstrap*.  In this post, I'll provide an introduction to cross-validation, and will touch on the boostrap in a future post.

Cross-validation is often used to estimate the *test error* associated with a certain fitted model in order to evaluate its performance, or to decide on an optimal level of flexibility for the model.  Evaluating a model's performance is often known as *model assessment*, while selecting the optimal level of flexibility is often known as *model selection*.

The structure of this post was influenced by the fifth chapter of *An Introduction to Statistical Learning: with Applications in R* by Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani.

### The Validation Set Approach

To evaluate a model's performance or select optimal flexibility, we are often interested in how the model performs on a test dataset.  That is, a dataset that was not used to train the model, and thus contains *new* data the model has not seen before.  In particular, we are often interested in selecting a model with low test error.  

Of course, we are interested in test error and not *training error* because training error can be deceiving and often underestimates test error.  Because the observations used to train the model are also used to calculate the training error, models with more flexibility that can better fit the nuances of the training data will tend to have lower training error.  We refer to this as *overfitting*, and overfit models will tend to perform poorly on new test data. 

A simple method to estimate test error is the *validation set approach*.  This approach involves randomly dividing the set of available observations into a training set and a *validation set* or *hold-out set*.  The model of interest is trained on the training set, and then is used to make predictions on the observations in the validation set. 

The resulting validation set error rate is typically assessed using mean squared error (MSE) in the case of a quantitative response or misclassification rate in the case of a qualitative response.  This validation set error rate provides an estimation for the test error rate.

A training-validation set split of 70%/30% or 80%/20% is commonly used, depending on the overall number of available observations.  

The validation set approach is conceptually quite simple and easy to implement, and is particularly useful for demonstration purposes, but has two potential drawbacks:

1. Due to the randomness of creating the training and validation sets, the validation set estimate of the test error is often highly variable.  A group of outliers, for example, may end up in the training set, the validation set, or split among both sets simply due to chance.

2. Since a subset of observations is intentionally left out of model training, the model is trained on less data.  Because models tend to perform better when more data is available, the resulting validation set error rate will tend to overestimate the test error rate.  If the same model were fit on all available observations, it would likely provide a lower test error rate in practice.

### Leave-One-Out Cross-Validation

Leave-one-out cross-validation (LOOCV) is a related method to the above described validation set approach, but it attempts to address that method's drawbacks.

Like the validation set approach, LOOCV involves splitting the observations into a training set and a validation set.  However, unlike the simple validation set approach, the validation set used in LOOCV is a single observation $(x_1, y_1)$ and the remaining observations ${(x_2, y_2), \ldots, (x_n, y_n)}$ make up the training set.  The statistical learning method of interest is then fit on the $n-1$ training observations, and a prediction $\hat{y}$ is made for the single excluded validation observation $x_1$.  Because $(x_1, y_1)$ was not used to fit the model, $MSE_1=(y_1 - \hat{y}_1)^2$ provides an approximately unbiased estimate for the test error.  Of course, even though $MSE_1$ is unbiased for the test error, it is based on a single observation and is thus highly variable.

To address this issue, we repeat this procedure by creating $n$ paired training and validation sets, where each validation set contains a single observations $(x_i, y_i)$.  For each validation set, we then calculate $MSE_i=(y_i - \hat{y}_i)^2$ such that we have $n$ values $MSE_1, MSE_2, \ldots, MSE_n$.  The LOOCV estimate of the test error is the average of these $n$ test error estimates:

$$
\begin{aligned} 
CV_{(n)} = \frac{1}{n} \sum_{i=1}^{n} MSE_i.
\end{aligned}
$$

LOOCV has a couple of advantages over the validation set approach.  First, the LOOCV test error estimate is calculated by repeatedly fitting the same statistical learning method to multiple training sets that contain $n-1$ observations.  Therefore, the LOOCV test error estimate is much less biased than that of the validation set approach, which is determined using only a subset of the available observations.  As a result of this, the LOOCV test error estimate tends not to overestimate the test error rate as much as that of the validation set approach.

Second, LOOCV will produce the same results every time, in contrast to the validation set approach which is inherently random in nature. 

However, LOOCV can be computation expensive, since the model has to be fit $n$ times.  In the special case of least squares linear or polynomial regression, we can make use of a handy shortcut to calculate the LOOCV test error estimate via a single model fit.  This formula holds:

$$
\begin{aligned} 
CV_{(n)} = \frac{1}{n} \sum_{i=1}^{n} (\frac{y_i-\hat{y_1}}{1-h_i})^2,
\end{aligned}
$$

where $\hat{y}$ is the $i$th fitted value from the original least squares fit, and $h_i$ is the leverage discuss in [this blog post](https://ethanwicker.com/2021-01-19-multiple-linear-regression-004/).

In general, LOOCV can be used with any kind of predictive model.  However, the above formula only holds in the case of least squares linear or polynomial regression.  In the general case, the model must be fit $n$ times to calculate the LOOCV test error estimate.

### $k$-Fold Cross-Validation

An alternative to LOOCV is $k$-fold cross-validation.  This method involves dividing the set of observations into $k$ groups, or *folds*, of approximately equal size.  The first fold is treated as a validation set, and the statistical learning method of interest is fit on the remaining $k-1$ folds.  The $MSE_1$ is then computed on the observations in the validation set, and this procedure is repeated $k$ times using a different fold as the validation set each time.  This procedure produces $k$ estimates of the test error $MSE_1, MSE_2, \ldots, MSE_k$.  The $k$-fold cross-validation test error estimate is then computed as the average of these values:

$$
\begin{aligned} 
CV_{(k)} = \frac{1}{k} \sum_{i=1}^{k} MSE_i.
\end{aligned}
$$

LOOCV is, of course, a special case of $k$-fold cross-validation in which $k=n$.

In practice, $k$-fold cross-validation is typically performed using $k=5$ or $k=10$.  The most obvious advantage of using 5 or 10 instead of $n$ as the value of $k$ is computational.  Instead of fitting the model $n$ times, the model need only be fit 5 or 10 times.

A less obvious but potentially more important advantage of $k$-fold cross-validation over LOOCV is that $k$-fold cross-validation often gives more accurate test error estimates.  This advantage has to do with the bias-variance tradeoff.  

Because LOOCV is fit using training sets containing $n-1$ observations, it is clear this method produces approximately unbiased estimates of the test error.  That is, a particular estimate of the test error is not *biased* by the random sample of observations used to calculate it.  In contrast, the validation set approach is more likely to produce a biased estimate of the test error, since only a subset of observations is chosen as the training set, and this subset is chosen randomly.  In the validation set approach, two randomly chosen training sets will likely produce two different test error estimates, and each estimate will be biased by the particular training set used.

As an extension, we can see that $k$-fold cross-validation will lead to an intermediate level of bias since each training set contains $(k-1)n/k$ observations - fewer than LOOCV but more than the validation set approach.  From the perspective of bias reduction, it is clear LOOCV is preferred over $k$-fold cross-validation.

However, when evaluating an estimating procedure, we must also consider the procedure's variance.  As such, it turns out that LOOCV has higher variance than $k$-fold cross-validation when $k<n$.  When performing LOOCV, we are in affect averaging the outputs of $n$ fitted models, each of which is trained on a nearly identical set of observations.  Therefore, the outputs are highly positively correlated with one another.  

In contrast, when performing $k$-fold cross-validation with $k<n$, we are averaging the outputs of $k$ fitted models that are somewhat less correlated with each other since the overlap between training sets is smaller.  Because the mean of meany highly correlated quantities has higher variance than does the mean of many quantities that are not as highly correlated, the test error estimate resulting from LOOCV will tend to have higher variance than does the test error estimate from $k$-fold cross-validation.

In summary, LOOCV will produce an unbiased estimate of the test error rate since the training sets contain nearly all observations.  However, the spread or variance of these test error rates will be lower when estimated via $k$-fold cross-validation with $k<n$ because of the averaging of the less correlated $MSE_i$ quantities.  Typically, we find the decrease in bias of LOOCV does not offset the increased variance, and thus $k$-fold cross-validation produces more accurate test error estimates.

### Cross-Validation on Classification Problems

In the above post, I've mostly considered cross-validation in the regression setting where the response $Y$ is qualitative.  As such, the measure of model performance has been mean squared error.  However, cross-validation is also appropriate for the classification setting where $Y$ is qualitative.  Instead of using MSE to quantify test error, we instead use the number of misclassified observations.  For example, in the classification setting, the $k$-fold cross-validation error rate is given by

$$
\begin{aligned} 
CV_{(k)} = \frac{1}{k} \sum_{i=1}^{k} \text{Err}_i
\end{aligned}
$$

where $\text{Err}_i = I(y_i \neq \hat{y)_i$.  Here $I()$ is an *indicator* function, which takes a value of 1 when $y_i$ does not equal $\hat{y)_i$, and 0 otherwise.
