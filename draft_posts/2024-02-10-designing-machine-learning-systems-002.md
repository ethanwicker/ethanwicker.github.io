---
layout: post
title: "10 more takeaways from Designing Machine Learning Systems"
subtitle: "Continual learning, model calibration, addressing data distribution shifts, and more"
comments: false
---

This is a second post detailing ___ more takeaways I had after recently reading Chip Huyen's [*Designing Machine Learning Systems*](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/).  You can find the [first post here](https://ethanwicker.com/2024-02-09-designing-machine-learning-systems-001/).

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
- [Testing in production](#testing-in-production)
- [The power of end-to-end Data Scientists](#the-power-of-end-to-end-data-scientists)
- [Understanding ML system trade-offs](#understanding-ml-system-trade-offs)


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

How does your model performance compare against the already established business solution?  Perhaps the existing solution is a collection of `if`/`else` statements, a semi-automated Excel notebook, or a third-party vendor.  Even if the model performance is inferior to the existing solution, if can still have high value if it's cheaper or easier to implement and maintain.

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

Note, once you have your critical slices, you'll still need sufficient, correctly labeled data to evaluate performance on each slice.

### Data distribution shifts

    - Data distribution shifts are due to issues cased by things other than underlying population chaning (p.230)
    -     - Addressing data distribution shifts: train with massive datasets, train w/o requiring new labels, retrain using labeled data (p.248-249)

### Degenerate feedback loops

    - Degerate feedback loops (p.233-236): how to detect and correct


### Continual learning

- continual learning (p.261, 264-270)
      - continual learning is about setting up infrastructure in a way that allows you to update your models whenever it is needed, whether from scratch or fine-tuning, and to deploy this update quickly (p.267)
      - two types of model updates (model iteration vs. data iteration)
      - continual learning is a superset of batch learning
      - feature reuse (log and wait) (p.277)
      - different triggers for model retraining (p.279)

### Determining how often to update deployed models

-     - How often to update your models (p.279)
      - value of data freshness --> run experiments
      - value of data freshness + trade-offs btw model iteration and data iteration
      - this is the only suggestion she gives

### Testing in production

-     - Testing in productions (p.280ish)
      - needs, test splits, testing in production
      - shadow deployments
      - a/b testing
      - canary release
      - interleaving experiments
      - bandit algorithms
  

### The power of end-to-end Data Scientists

I agree with Chip so much on this topic - I was thrilled to see her include it.

In many, many organizations, end-to-end Data Scientists should be the default mode of operation, as opposed to creating teams full of specialized data engineers, data scientists, and machine learning engineers.  End-to-end Data Scientists are often referred to as generalists or full-stack data scientists.

In short, teams full of specialists have increased communication and iterate and learn slower.  In contrast, a Data Scientist who can manage the entire end-to-end process of understanding the business problem, organizing the data, training the model, and measuring and managing the production deployment can be much more effective and allow the organization to learn and iterate quicker.  The full-stack data scientist also gains greater context for the business, it's problems, and how to effectively solve them.

For this to be possible, the underlying data infrastructure must be strong.  The platform team must have built robust tools.  Assuming this is the case, a data scientist who can make use of the tooling across the pipeline will have an outsized impact.  Stitch Fix and Eugene Yan each have great articles on this topic [here](https://multithreaded.stitchfix.com/blog/2019/03/11/FullStackDS-Generalists/) and [here](https://applyingml.com/resources/end-to-end/).

Eric Colson of Stitch Fix makes it clear that we should be optimizing our data science teams for productivity gains.  The goal is not execute, but is instead to "learn and develop profound new business capabilities."  On creating robust data infrastructure, he says

> It is important to note that this amount of autonomy and diversity in skill granted to the full-stack data scientists depends greatly on the assumption of a solid data platform on which to work. A well constructed data platform abstracts the data scientists from the complexities of containerization, distributed processing, automatic failover, and other advanced computer science concepts

### Understanding ML system trade-offs


    - optimizing for one metric does not hold all others static (p.349)
      - privacy vs. accuracy trade off
      - compactness vs. fairness trade off (p.350)
