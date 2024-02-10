---
layout: post
title: "Five things I learned from Chip Huyen's Designing Machine Learning Systems"
subtitle: Some tidbits of knowledge (or maybe "Active learning, detecting data leakage, and more")
comments: false
---

### Big Header

#### Small Header

I recently read through Chip Huyen's (*Designing Machine Learning Systems*)[https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/].  Some of these I was aware of, but appreciated Chip's review.

I should like to the pages in the book where I get my info.

These items are in no particular order and I'm not including details here and there I'm familair with.  may or may not be good for your reading. (but I should try to make clear and concise)

Notes:
    - repository for storing structured data is data warehouse, respository for storing unstructured data is data lake (p.67).  Combining the flexibility of data lakes and data management of data warehouse = hybrid solution = data lakehouse (both Databricks and Snowflake offer) (p.72)
    - Batch processing is a special case of stream processing (p.79)
    - Consider creating in-house labeling teams (p.88)
    - Weak supervision (p.95)
    - Semi-supervised learning (p.98) --> class method is self-training.
    - Active learning (p.101)
    - With class imbalances, might want to train to be better at predicting 95th percentile (p.103)
    - The precision-recall curve to understand class imbalances (p.108)
    - Oversampling and undersampling
      - Popular method of undersampling low-dimensional data = Tomek links (p.109)
      - Popular method of oversampling low-dimensional data is SMOTE (p.109)
      - more sophisticated resampling methods such as near-miss and one-sided selection are infeasible/expensive in high-dimensional data (p.110)
      - two phase learning and dynamic sampling (p.110)
      - class-balanced loss. focal loss, ensembles have been shown to be helpful

# START HERE FOR SECOND POST

    - The hashing trick to encode unknown amount of categorical fields (p.13-132)
    - Splitting time-correlated data by time instead of randomly to prevent data leakage  --> **these correlations are not always obvious** (p.136)
    - group leakage (p.139)
    - Detecting data leakage (p.140)
      - measure predictive power of each feature wrt target variable
      - ablation studies
      - keep an eye out for new features added to your model
      - in general, be very careful looking at test splits, if ever
    - Neural architecture search (p.175)
    - The five types of baselines (p.180)
    - Pertubation tests (p.182)
    - model calibration (p.183)
    - slice-based evaluation (p.185)
      - How to determine slices? (p.188) heuristic-based, error analysis, slice finder
      - once discovered critical slices, will need sufficient, correctly labeled data for each of these slices for evaluation
    - one benefit of online productions is you don't have to generate a prediction for all users, especially when users don't use your product daily (p.201)
      - Chip thinks as hardware becomes more specialized and powerful and techniques improve, online prediction might become the default
      - unifying stream and batch processing has become popular topic recently (p.205)
    - Different ways of doing model compression (p.206-208)
      - low rank factorization
      - knowledge distillation
      - pruning (can introduce bias)
      - quantization
    - Pros and cons of deploying models on the edge (p.212-213)
    - Data distribution shifts are due to issues cased by things other than underlying population chaning (p.230)
    - Degerate feedback loops (p.233-236): how to detect and correct
    - Addressing data distribution shifts: train with massive datasets, train w/o requiring new labels, retrain using labeled data (p.248-249)
    - four major concerns when doing feature monitoring (p.254-255)
    - log monitoring at scale should be done via streaming, not batch (p.257)
    - monitoring vs. observability (p.250)
    - ML observability contains interpretability (p.261)
    - continual learning (p.261, 264-270)
      - continual learning is about setting up infrastructure in a way that allows you to update your models whenever it is needed, whether from scratch or fine-tuning, and to deploy this update quickly (p.267)
      - two types of model updates (model iteration vs. data iteration)
      - continual learning is a superset of batch learning
      - feature reuse (log and wait) (p.277)
      - different triggers for model retraining (p.279)
    - How often to update your models (p.279)
      - value of data freshness --> run experiments
      - value of data freshness + trade-offs btw model iteration and data iteration
      - this is the only suggestion she gives
    - Testing in productions (p.280ish)
      - needs, test splits, testing in production
      - shadow deployments
      - a/b testing
      - canary release
      - interleaving experiments
      - bandit algorithms
    - Data scientists have bias when evaluating their own models (p.290)
    - when choosing a model deployment service (examples here), check whether this service makes it easy for you to perform the tests that you want (p.321)
    - feature store = overloaded term --> three things: feature management, feature transformation/computer, feature consistency (p.325-326)
    - highly beneficial for an ML system to have SME's involved in the entire lifecycle (p.335)
    - End-to-end DSs have immense power -> link to two pages --> but they need proper infrastructure to support them (p.___)
    - Transparency is the first step in building trust in systems (p.343)
    - Biases can creep in your system through the entire workflow.  first step is to discover how (p.347)
      - model object comment on p.348
      - evaluation on p.348
    - optimizing for one metric does not hold all others static (p.349)
      - privacy vs. accuracy trade off
      - compactness vs. fairness trade off (p.350)