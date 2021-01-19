---
layout: post
title: "Multiple Linear Regression #3"
subtitle: Qualitative Predictors, Interaction Terms, and Non-linear Relationships
comments: false
---

This post is the third in a series on the multiple linear regression model.  In previous posts, I introduced the multiple linear regression model and explored a comparison of Python's scikit-learn and statsmodels libraries.  However, both of these previous posts exclusively explored quantitative predictors.  

In this post, I'll explore qualitative predictors.  I'll also explore classical methods of relaxing some restrictive assumptions of the linear model - namely the *additive* and *linear* assumptions - via the use of *interaction terms* and *polynomial regression*.

This structure of this post was influenced by the third chapter of *An Introduction to Statistical Learning: with Applications in R* by Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani.

#### Qualitative Predictors

Depending on the context, qualitative predictors are sometimes referred to as categorical or factors variables.  The most common methods of implementing these predictors in a linear model is via *indicator* or *dummy variables*.  Below, I'll introduce this topic in the context of qualitative predictors with only two *levels*, where a level refers to a unique value the qualitative variable can take.  I'll then extend the concept to qualitative predictors with multiple levels.

##### Qualitative Predictors with Two Levels

In the case where we have a predictor variable with two possible levels, it is straightforward to create a dummy variable that takes on two possible numerical values, $0$ and $1$:

$$
\begin{aligned} 
x_i = 
    \begin{cases}
        1 & \text{if $ith$ observation is the first level}\\
        0 & \text{if $ith$ observation is the second level}
    \end{cases} 
\end{aligned}
$$.

We then use this dummy variable in the linear regression model:

$$
\begin{aligned} 
Y_i = \beta_0 + \beta_1X_i + \epsilon_i = 
    \begin{cases}
        \beta_0 + \beta_1 + \epsilon_i & \text{if $ith$ observation is the first level}\\
        \beta_0 + \epsilon_i & \text{if $ith$ observation is the second level}
    \end{cases}
\end{aligned}
$$.

Alternatively, instead of encoding our dummy variable as $0$ or $1$, we could instead have encoded it as $-1$ or $1$.  Doing so would have changed the coefficient estimates of the model, and the interpretation, but not the predictors.  In addition, when performing statistical inference, we interpret the corresponding p-values of qualitative predictors exactly as we do quantitative predictors.

##### Qualitative Predictors with More than Two Levels

In the case where a qualitative predictor has more than two levels, we cannot use a single dummy variable to represent all possible levels.  In this situation, we simply create additional dummy variables.  For example, if we have a predictor variables with three possible levels, we can encode these levels into two dummy variables

$$
\begin{aligned} 
x_{i1} = 
    \begin{cases}
        1 & \text{if $ith$ observation is the first level}\\
        0 & \text{if $ith$ observation is not the first level}
    \end{cases} 
\end{aligned}
$$

and 

$$
\begin{aligned} 
x_{i2} = 
    \begin{cases}
        1 & \text{if $ith$ observation is the second level}\\
        0 & \text{if $ith$ observation is not the second level}
    \end{cases} 
\end{aligned}
$$.

We then use these dummy variables in the regression equation to obtain the model

$$
\begin{aligned} 
Y_i = \beta_0 + \beta_1X_{i1} + \beta_2X_{i2} + \epsilon_i = 
    \begin{cases}
        \beta_0 + \beta_1 + \epsilon_i & \text{if $ith$ observation is the first level}\\
        \beta_0 + \beta_2 + \epsilon_i & \text{if $ith$ observation is the second level}
        \beta_0 + \epsilon_i & \text{if $ith$ observation is the third level}
    \end{cases}
\end{aligned}
$$.

There will always be one fewer dummy variables than the number of levels of the predictors.  The level with no dummy variable - the third level in the above example - is referred to as the *baseline*.  Of note, depending on the context, dummy variable encoding may also be referred to as *one-hot encoding*.  These techniques are equivalent, but one-hot encoding tends to keep the *baseline* level as an encoded variable, as opposed to dropping it.  This is sometimes preferred in machine learning models using *regularization*, which will be discussed in a future post.

With the dummy variable approach, we can incorporate both quantitative and qualitative predictors into the multiple regression model.  Graphically, this results in parallel hyperplanes in the predictor space.  As mentioned above, the interpretation of the p-values does not change for these qualitative predictors, but the p-values themselves do depend on the choice of dummy variable encoding.  However, we can still use the F-test to test $H_0: \beta_1 = \beta_2 = \ldots = \beta_p = 0\$ to determine if any relationship exists between the predictors and the response.

(Maybe above: \{0,1,2,\,\ldots\})

#### Extensions of the Linear Model

The standard linear model provides interpretable results and has been shown to perform well on many real-world problems.  However, it makes several strict assumptions that are often not true in practice.  Two such assumptions are that the relationship between the predictors and response are additive and linear.  In future posts, I'll explore more sophisticated methods of relaxing these assumptions, but below I'll introduce two classical strategies.

##### Removing the Additive Assumption: Interaction Terms

In the standard linear regression model with two predictor variables

$$
\begin{aligned} 
Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \epsilon 
\end{aligned}
$$

we interpret a one unit increase in $X_1$ as being associated with an average $\beta_1$ increase in $Y$.  Notice, $X_2$ has no effect on this interpretation.  In practice however, it might be more reasonable to assume that a change in $X_1$ is associated with a change in $X_2$.  As such, we can extend the standard linear regression model by including a third predictor, called an *interaction term*, which is the product of $X_1$ and $X_2$.  This results in the model 

$$
\begin{aligned} 
Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + + \beta_3_X_1X_2 + \epsilon \\
  = \beta_0 + (\beta_1 + \beta_3_X_2)X_1 + \beta_2X_2 + + \epsilon
\end{aligned}
$$.

Note, with the new interaction term, a change in either $X_1$ or $X_2$ will affect the impact the other predictor has on $Y$.  Thus, they no longer have a simple additive relationship.

Just as before, when doing statistical inference, we interpret the p-value of the interaction term as we would any other term.  If the p-value of the interaction term is significant, we have strong evidence that the true relationship is non-linear.

In the event an interaction term is significant, but the *main effects* are not, we should still include the main effects via the *hierarchy principle*.  Because $X_1X_2$ tends to be correlated with $X_1$ and $X_2$, leaving them out tends to alter the meaning of the interaction.

The concept of interaction terms is also applicable to qualitative variables, or a combination of qualitative and quantitative variables.  In particular, an interaction between a qualitative and quantitative variables has a particularly nice interpretation.  Instead of two parallel hyperplanes, we get two intersecting hyperplanes.

##### Removing the Linear Assumption: Non-linear Relationships

In future posts, I'll explore more sophisticated methods of modeling non-linear relationships in general settings.  However, below, I'll briefly introduce *polynomial regression*, which is a simple method to extend the linear model to capture non-linear relationships.  

Polynomial regression can capture non-linear associations by including transformed versions of the predictors in the model.  For example, a quadratic regression may take the form

$$
\begin{aligned} 
Y = \beta_0 + \beta_1X_1 + \beta_2X_1^2 + \epsilon 
\end{aligned}
$$.

It is important to state that this quadratic regression model is still linear.  It is simply the standard linear model where $X_2 = X_1^2$.  As you can imagine, cubic, quartic, and other higher degree polynomial regressions can be achieved via the same variable transformation method.

While polynomial regression can be a useful technique, it does have some drawbacks.  In particular, because the end behavior of polynomials tends to vary extremely, the models can perform poorly when extrapolating.
