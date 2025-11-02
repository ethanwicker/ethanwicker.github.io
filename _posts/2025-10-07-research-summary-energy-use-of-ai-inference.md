---
layout: post
title: "Research summary"
subtitle: "Energy Use of AI Inference: Efficiency Pathways and Test-Time Compute"
comments: false
---

# Summary

- Paper: [Energy Use of AI Inference: Efficiency Pathways and Test-Time Compute](https://arxiv.org/abs/2509.20241)
- Published by Microsoft researchers on September 24, 2025

## Background

AI energy requirements are commonly split into two categories:

- *Energy for training* → the energy required to train, or create the AI model  
- *Energy for inference* → the energy required to query, or use the AI model  
  - ("inference" is a carryover from statistics, where you "infer" a prediction from a statistical model)

Inference is sometimes split into **test-time** and **non-test-time**:

- *Test-time energy* refers to the energy required for a reasoning model that "loops" over itself (e.g., agentic workflows)  
  - ("test-time" is a carryover from machine learning, where you literally test the quality of your prediction or classification model)

## Context

This paper made a bit of a splash because the authors claimed most current AI inference energy estimates are **4–20x too high** ⭐

- They only looked at inference energy needs. They did not consider training energy needs  
- They looked at current inference needs, as well as provided estimates for likely energy efficiency improvements  
- They focused on large-scale production deployments (as opposed to smaller scale, less efficiency deployments)

## What they did

- The authors created a formula to estimate the energy used for a single GPU server  
- This formula used things like the number of input and output tokens, the known power draw of GPUs, and the known "power usage effectiveness" of AI data centers to estimate the energy needed to get some output  
- They then ran many simulations over five open-source AI models:
  - DeepSeek-R1 671B  
  - Llama 3.1 405B  
  - Llama 3.1 Nemotron Ultra 253B  
  - Mixtral 8x22B  
  - Llama 3.1 70B

From this, they created istributions of the energy requirements for a single query through one of these models.

They also did some estimation to determine near-future energy efficiency gains in three areas:

- *Model improvements* → creating more energy efficient models (such as MoE architectures, distillation, quantization)  
- *Workload management improvements* → better ways to optimize model deployments (such as improved caching methods or model routing methods)  
- *Hardware and datacenter improvements* → improving chip and datacenter design (such as cooling techniques and switching to custom AI inference chips)

## Findings

- Current energy estimates based on small-scale or non-optimized deployments overestimate the energy use of inference by **4–20x**
- Currently, for large frontier models running on state-of-the-art hardware under realistic workloads, the authors estimate a **median energy per query of 0.34 Wh** (equivalent to streaming Netflix for ~16 seconds)
- Extending this to **agentic workflows**, the median energy for a query increases **13x to 4.32 Wh** (equivalent to streaming Netflix for ~3.5 minutes)
- Currently, the authors estimate it takes **1.8 GWh to serve 1 billion queries** (assuming 10% of queries are for agentic workloads, 90% for non-agentic workloads)
  - This is ~6x higher than the energy required to perform 1 billion daily web searches (~0.3 GWh)
  - This is roughly equivalent to ~1% of YouTube’s daily energy demands  
  - For context, ChatGPT serves ~2.5 billion queries per day
- The authors see “plausible” energy gains of **8–20x** (from a combination of model improvements, workload management improvements, and hardware and datacenter improvements)

## Takeaways

- Most energy estimates are 4–20x too high  
- **Why?** They are based on unrealistic assumptions or extrapolate from low-quality, poorly optimized deployments  
- The median energy per query is equivalent to streaming Netflix for 16 seconds ⭐  
- Currently, AI inference is roughly 6x more energy intensive than web search  
- AI inference will likely get 8–20x more efficient in the next few years. Conservatively, we’ll see 2x or 4x improvements  
- We should of course be aware and considerate about the AI energy footprint, but we shouldn’t be alarmist. The authors include two historical projections, as counterexamples to the alarmist narrative:  
  - A 1999 forecast warned that the internet could consume 50% of the US grid by 2010, but actual usage peaked around 2%  
  - A 2015 forecast estimated data centers would consume 8,000 TWh per year by 2030 (more than 25% of global electricity), but updated estimates now suggest 600–800 TWh per year (about 2% of global electricity)

In the authors’ opinions:  
> “The current surge in AI inference demand may seem unprecedented, but it follows the same historical pattern: while absolute energy use will grow, efficiency gains in hardware, software, and deployment strategies at scale can moderate its long-term energy footprint.”

## tl;dr

- AI energy usage for using (not training) models is vastly overstated (by 4–20x)  
- Even so, we will see large improvements in energy efficiency. How large? Hard to say, but between **4–10x** is likely (the authors say 8–20x improvements)  
- The median energy per query to an LLM is equivalent to streaming Netflix for 16 seconds ⭐  
- Agentic workflows are 10–15x more energy intensive. We’ll see progress here, but there’s no denying they do take more energy  
- ChatGPT serves ~2.5 billion queries per day. That is ~3% of all of YouTube’s energy requirements for an entire day  
- Yes, AI uses energy. No, in 2025, it’s not that crazy.  
- **Caveat #1:** This stuff gets complex fast. Lots of energy is used to mine the rock to eventually make it into a GPU and ask some LLM questions about calculus. The numbers above only relate to asking some LLM questions about calculus  
- **Caveat #2:** AI training energy usage is a whole different topic, worthy of its own post  

## Final thoughts

All of this begs one question: *Is this energy use too much?*

I can’t answer that. But if it is too much, then so is scrolling social media, or using Google Maps too often, or playing online games. Using AI is roughly in the same energy ranges.  
We never hear the narrative that Instagram is too energy intensive, even though it’s serving billions of posts a day to millions of users, also via deep learning models.
