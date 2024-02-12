---
layout: post
title: "11 more takeaways from Designing Machine Learning Systems"
subtitle: "Continual learning, model calibration, distribution shifts, trade-offs and more"
comments: false
---

This is a second post detailing 11 more takeaways I had after recently reading Chip Huyen's [*Designing Machine Learning Systems*](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/).  You can find my [first post here](https://ethanwicker.com/2024-02-09-designing-machine-learning-systems-001/).

In no particular order, here are some more insights I found helpful.

- [Five types of model baselines](#five-types-of-model-baselines)
  - [Random baseline](#random-baseline)
  - [Simple heuristic](#simple-heuristic)
  - [Zero rule baseline](#zero-rule-baseline)
  - [Human baseline](#human-baseline)
  - [Existing solutions](#existing-solutions)
- [Model calibration](#model-calibration)
- [Slice-based evaluation](#slice-based-evaluation)
- [Data distribution shifts](#data-distribution-shifts)
- [Degenerate feedback loops](#degenerate-feedback-loops)
- [Continual learning](#continual-learning)
- [Determining how often to update deployed models](#determining-how-often-to-update-deployed-models)
- [The power of end-to-end Data Scientists](#the-power-of-end-to-end-data-scientists)
- [Understanding ML system trade-offs](#understanding-ml-system-trade-offs)
  - [Privacy vs. accuracy](#privacy-vs-accuracy)
  - [Compactness vs. fairness](#compactness-vs-fairness)


### Five types of model baselines

When deciding how to approach a problem and which methods to explore, it's essential to understand how well simpler methods perform.  If we're trying to predict student performance on a particular assessment, do we even need a machine learning-based solution?  Would a heuristic-based solution work just as well?  If a trained model does offer improvements over simpler solutions, how much greater are the improvements?  Do they warrant the time and resources to train and maintain the model?

Without a baseline, we really can't answer these questions.  Of course, exact baselines and how they are defined will vary between use cases.  In *Designing Machine Learning Systems*, Chip shares five baselines that might work well across many use cases.

#### Random baseline

For a classification task, a random baseline refers to the system's performance if we just randomly assign class labels.  The labels could be randomly drawn from the uniform distribution or the task's label distribution.  

For example, if we want to predict whether a student will need additional support to solve a problem, and if in our training data 10% of students do need additional support and 90% do not, we can just randomly assign labels from this distribution.  If we did this, our $F_1$ score would be 0.1 and our accuracy would be 0.82.  Of course, ideally our trained classifier would have higher $F_1$ and accuracy values on the test set, but we can't be certain unless we establish the random baseline.

#### Simple heuristic

Using a simple heuristic as a baseline is use case specific, but the idea is that instead of using machine learning, what if we just used some other simple rule or procedure to create our predictions?  For example, if we're building a course recommendation service for an online learning platform, we could just serve our users the five most popular courses they have not taken, sorted in decreasing order of popularity.  If we did this, how likely is a user to start or complete one of the recommended courses?

#### Zero rule baseline

The zero rule baseline refers to just always predicting the most common class and serving that to our users.  For a low-code educational programming tool like [Scratch](https://scratch.mit.edu/), we might be interested in recommending the next icon to new users to scaffold their learning on practice exercises.  The simplest model would be to just recommend the overall most commonly used icon each time.  Of course, a better solution would instead recommend icons that are actually useful to the practice exercise.  

#### Human baseline

For many, many problems, we're interested in automating tasks at or above human-level performance, so it's helpful to know how well humans actually perform.  For the use case of generating automated essay feedback, does the machine-learning solution perform as well as the human grader?  Even if the system isn't meant to entirely replace humans, it can still be important to understand in what scenarios the system has near or above human-level performance.  When generating automated essay feedback, perhaps the system does great on simpler essays more commonly assigned to fifth-grade students but does not perform well on more complex essays written by high school students.

#### Existing solutions

How does your model performance compare against the already established business solution?  Perhaps the existing solution is a collection of `if`/`else` statements, a semi-automated Excel notebook, or a third-party vendor.  Even if the model performance is inferior to the existing solution, it can still have high value if it's cheaper or easier to implement and maintain.

### Model calibration

Model calibration refers to the ability of a classification model to provide accurate probability estimates.  More specifically, if a model predicts some outcome with 90% probability, that outcome should actually occur around 90% of the time.  A well-calibrated model has this property, while a poorly-calibrated model does not.  

To add a bit more detail, a model is often designed to output a probability for each possible class label.  For a deep learning model, this is often done via a softmax layer.  The model might predict the labels for class 1, class 2, and class 3 with probabilities $p_1 = 0.40$, $p_2 = 0.55$ and $p_3 = 0.05$, respectively.  In practice, we would typically take the model's highest outputted probability as the predicted label, which would be $p_2$ in this case.  For a well-calibrated model, the values of $p_1$, $p_2$ and $p_3$ would reflect the true probabilities of these classes occurring.

Model calibration is most important when we care not only about the predicted labels but also the estimated uncertainties.  For example, a well-calibrated model might help a teacher diagnose a student's conceptual misunderstandings more quickly.

There are a few methods to improve model calibration, including Platt scaling and isotonic regression.  See [this Medium article](https://medium.com/@heinrichpeters/model-calibration-in-machine-learning-29654dfcef43) by Heinrich Peters for more details.

### Slice-based evaluation

Slice-based model evaluation refers to separating your data into subsets and analyzing your model's performance on each subset separately.  In practice, organizations often prioritize overall metrics like $F_1$ or accuracy across the entire dataset, as opposed to calculating and analyzing these metrics on individual slices of data.

This over-prioritization of overall metrics can lead to two problems:

1. Your model performs differently on different slices of data when the model should perform the same.
2. Your model performs the same on different slices of data when it should perform differently.

For the first problem, we might want our model to have the same degree of accuracy across each slice.  For example, maybe we expect our model to correctly predict student academic performance across different genders or races, but we might find it is less accurate on minority groups.  In this case, we might prefer using a different model that is equally accurate across all slices, even if it is slightly less accurate overall.

For the second problem, we might want our model to perform better on more critical slices.  As an example of this, perhaps we train a model to predict which school districts are more likely to stop using a product.  We'd likely want our model to be more accurate on school districts that have heavy adaption of a product, as opposed to those that are just exploring using the product, since the heavy adaption school districts generate more revenue.

To determine which slices are critical to explore, Chip recommends three approaches:

1. Heuristic-based: Slice your data via domain knowledge
2. Error analysis: Manually go through misclassified examples to find patterns among them
3. Algorithmic: Using a tool like [Slice Finder](https://ieeexplore.ieee.org/abstract/document/8731353), or generating slice candidates via beam search or clustering, and then pruning away clearly bad slice candidates 

Note: Once you have your critical slices, you'll still need sufficient, correctly labeled data to evaluate performance on each slice.

### Data distribution shifts

Data distribution shifts, sometimes referred to as data drift, refer to a collection of phenomena where the data used to train a model no longer matches the data the model is deployed over.  Data distribution shifts can occur in a variety of ways, including *covariant shift*, *label shift*, and *concept drift*.  These distribution shifts are typically only a problem if they degrade your model's performance, but a sudden unexpected increase in model performance should also be investigated.

To detect data distribution shifts, it's helpful to have access to labels within a reasonable amount of time.  However, in industry, most data drift detection methods focus on detecting changes in the input distribution.  Statistical methods can be helpful for this, including two-sample hypothesis tests such as the Kolmogorov-Smirnov test (K-S test).  Other statistical methods such as Least-Squares Density Difference or Maximum Mean Discrepancy can also be helpful.  Reducing the dimensionality of your data before performing a two-sample test can also be helpful.

Temporal methods, such as time-series decomposition or time-series-based anomaly detection methods can also be used to detect data distribution shifts.

There are a few approaches to address data distribution shift challenges.  The first is to train on massive datasets, hoping that your model can learn such a comprehensive distribution that whatever data points your model will encounter in production will likely come from this distribution.  A second approach is to adopt a trained model to a target distribution without requiring new labels.  There have been a few proposed methods of doing this, but this is not common in industry currently.  Lastly, a third approach is to retrain your model using labeled data from the target distribution.   This is the approach most commonly used today.  With this approach, you can either retrain your model from scratch on both the old and new data or continue training the existing model on new data, a process called fine-tuning. 

It's important to note that the majority of what might look like data shifts on monitoring dashboards are caused by internal errors, such as bugs in the data pipeline, missing values incorrectly inputted, inconsistencies between the features extracted during training and inference, wrong model version, or bugs in the app interface that affect user behavior.  The data put into your model might be altered, but it might not always be because the true underlying population has changed.

### Degenerate feedback loops

A degenerate feedback loop occurs when a system's outputs themselves influence the system's future input.  These loops are especially common in recommendation or ad click-through-rate prediction systems.  Specifically, they occur when the system's predictions, such as course recommendations, influence how users interact with the system.  Since this new information is often used as training data for the next model iteration, this feedback loop can have unintended consequences.  Degenerate feedback loops have also been termed *expose bias*, *popularity bias*, *filter bubbles*, and *echo chambers*.

Detecting degenerate feedback loops can be challenging, especially when your system is offline and not serving results to users (because the loop is created via user feedback).  For recommendation systems in particular, it's possible to detect degenerate feedback loops by measuring the popularity diversity of a system's outputs even when the system is offline.  An item's popularity can be measured based on how many times users interact with it.  Since most items will be rarely interacted with, The popularity of all items will tend to follow a long-tail distribution.  Proposed metrics such as *aggregate diversity*, *average coverage of long-tail items*, and *hit-rate-against-popularity* can be used to quantify how homogeneous your system's outputs are.  

There are a variety of methods to correct degenerate feedback loops, including randomization and using positional features.  Randomization refers to instead of only showing users items that have been highly ranked for them, we show users random items and use their feedback to determine the true quality of these items.  Randomization can improve diversity but at the cost of user experience.  Contextual bandits offer a more intelligent exploration strategy over randomization, while [Schnabel et al.](https://arxiv.org/pdf/1602.05352.pdf) have shown that randomization and causal inference techniques can correct degenerate feedback loops in recommendation systems.

Using positional features to correct degenerate feedback loops refers to encoding positional information about the recommendations into features used by future model iterations.  For example, you might encode a boolean feature specifying if a recommendation was in the top position shown to users, or you might encode the actual numerical positioning of the recommendation.  Including these features allows you model to learn how an item's position influences user interactions.  To predict how users interact with your recommendations without the positional bias, the positional features can be adjusted, so that we can understand the true preference of various recommendations regardless of their position.

### Continual learning

Continual learning is an approach that prioritizes updating a model regularly, typically in micro-batches as new data comes in, as opposed to retraining the model from scratch.  Continual learning has many advantages, including combatting data distribution shifts and adapting to rare events.  Continual learning can also help against the continuous cold start problem, where your model has to make predictions for a new user without any historical data.  If your model can't adapt quickly, it won't be able to make recommendations relevant to new users until the next time the model is updated.

Continual learning, when set up properly, is really a superset of batch learning, as it supports all features of traditional batch learning.  Chip believes that continual learning is about setting up infrastructure to support updating and deploying models - whether from scratch or fine-tuning - whenever that is needed.

There are two types of iterations you might want to make to your system: *model iterations* and *data iterations*.  Model iterations refer to adding a new feature to an existing model architecture or altering the model architecture.  Data iterations refer to keeping the model architecture and features the same, but refreshing the model with new data.  Currently, stateful continual learning is mostly applied for data iterations.

Continual learning has a variety of challenges.  One challenge is getting access to fresh, labeled data.  To update our models, we need fresh data, and that data has to be available in the data warehouse.  An alternative to this is to pull data before it's deposited into the data warehouse (e.g., directly from the Kafka or Kinesis real-time transport).  If your model needs labeled data as well, this can be a bottleneck, so the best candidates for continual learning are tasks where natural labels can be assigned with short feedback loops.  Examples of these include dynamic pricing, stock price predictions, estimated time of arrival, and recommender systems.

Another challenge of continual learning is to make sure your new model is good enough to be deployed and replace your existing model.  Catastrophic forgetting is a specific problem in this space, and model evaluation itself does take time, which can be another bottleneck.

A third challenge of continual learning is algorithmic-based.  Specifically, some model types, such as neural networks perform better with micro-batch updating.  Other algorithms such as matrix-based or tree-based can be more challenging to adapt to the continual learning paradigm.  There have been tree-based algorithms that can learn from incremental amounts of data, most notably Hoeffding Tree and its variants Hoeffding Window Tree and Hoeffding Adaptive Tree.

To trigger your model update, there are four possible triggers:
- time-based - e.g., every 5 minutes
- performance-based - e.g., whenever model performance decreases
- volume-based - e.g., whenever you have access to enough new labeled data
- drift-based - e.g., whenever a major data distribution shift is detected

### Determining how often to update deployed models

Figuring out how often to update deployed models is a classic machine learning engineering challenge.  Most organizations update their models on some reoccurring schedule, likely determined via domain expertise or infrastructure limitations.  In contrast, a better method might be to evaluate how much value we get from fresh data.  One way to do this is to train your model on multiple past time windows, and then to evaluate its performance on today's data to see how performance changes.

Of course, we also need to consider if we should be updating our model architecture or just its training data.  If you find refreshing your model with new data has marginal performance gains, you should explore other architectures.  In contrast, if updating your model on new data does increase performance and is much cheaper and simpler to do, it might be worthwhile to delay updating the model architecture.

### The power of end-to-end Data Scientists

I agree with Chip so much on this topic - I was thrilled to see her include it.

In many, many organizations, end-to-end Data Scientists should be the default mode of operation, as opposed to creating teams full of specialized data engineers, data scientists, and machine learning engineers.  End-to-end Data Scientists are often referred to as generalists or full-stack data scientists.

In short, teams full of specialists have increased communication and iterate and learn slower.  In contrast, a Data Scientist who can manage the entire end-to-end process of understanding the business problem, organizing the data, training the model, and measuring and managing the production deployment can be much more effective and allow the organization to learn and iterate quicker.  The full-stack data scientist also gains greater context for the business, it's problems, and how to effectively solve them.

For this to be possible, the underlying data infrastructure must be strong.  The platform team must have built robust tools.  Assuming this is the case, a data scientist who can make use of the tooling across the pipeline will have an outsized impact.  [Stitch Fix](https://multithreaded.stitchfix.com/blog/2019/03/11/FullStackDS-Generalists/) and [Eugene Yan](https://applyingml.com/resources/end-to-end/) each have great articles on this topic.

Eric Colson of Stitch Fix makes it clear that we should be optimizing our data science teams for productivity gains.  The goal is not execution, but is instead to "learn and develop profound new business capabilities."  On creating robust data infrastructure, he says:

> It is important to note that this amount of autonomy and diversity in skill granted to the full-stack data scientists depends greatly on the assumption of a solid data platform on which to work. A well constructed data platform abstracts the data scientists from the complexities of containerization, distributed processing, automatic failover, and other advanced computer science concepts

### Understanding ML system trade-offs

An important concept not often discussed is that when optimizing our machine learning system for one metric, we almost always degrade another metric.  This is worth discussing in more depth, especially when one of the trade-offs might be privacy, bias, or fairness.

####  Privacy vs. accuracy

Privacy in machine learning and artificial intelligence is related to ensuring that trained models do not compromise the personal information of individuals.  There are multiple methods to increase privacy, including adding noise to training data or model predictions (differential privacy), or training models across multiple decentralized devices (federated learning).  On the contrary, model accuracy is often highly related to how accurate the input data is to the model.  Balancing this trade-off can be challenging, and is often domain or context-specific.

#### Compactness vs. fairness

Model compression and compact can be crucially important for edge deployment and inference speed, but by reducing model size we also typically reduce the model to a less complex representation of the real world.  Specifically, even though a compressed model might have similar overall accuracy to its non-compressed version, it might be less accurate across important data slices and reinforce biases.  For example, it might be less able to detect dark-toned faces or provide quality automated feedback for English language learners.