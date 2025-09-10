---
layout: post
title: "Research summary"
subtitle: "Your Brain on ChatGPT: Accumulation of Cognitive Debt when Using an AI Assistant for Essay Writing Task"
comments: false
---

# Summary

- Paper: [Your Brain on ChatGPT: Accumulation of Cognitive Debt when Using an AI Assistant for Essay Writing Task](https://www.media.mit.edu/publications/your-brain-on-chatgpt/)
- Published June 10, 2025 (pre-print)
- Published by researchers at MIT, Wellesley College, Massachusetts College of Art and Design

## Context

- This paper has been referenced multiple times in opinion news articles about AI in education ([here](https://time.com/7295195/ai-chatgpt-google-learning-school/),  [here](https://www.nytimes.com/2025/07/03/opinion/aritificial-intelligence-education.html), and [here](https://www.nytimes.com/2025/07/18/opinion/ai-chatgpt-school.html))
- It came up on a [daily AI podcast](https://open.spotify.com/episode/4ZHxqcy4YqmISQmoqaFVLJ?si=c317e6bd651a4003) I listen to, catalyzing me to finally read it

## Methods

- Researchers had 54 college students write 3 essays over 3 sessions (1 essay per session)
- Participants were divided into 3 groups:
  - LLM only (ChatGPT using gpt-4o)
  - Search engine only
  - Brain only (no LLM or search engine use)
- Participants chose from a collection of SAT essay prompts, and had 20 minutes to write their essay response
- Researchers used "electroencephalography (EEG) to record participants' brain activity in order to assess their cognitive engagement and cognitive load"
  - EEG = diagnostic test that measures the brain's electrical activity by placing small electrodes on the scalp
- Participants also completed interviews after each session, answering questions like:
  - *Can you summarize the main points or arguments you made in your essay?*
  - *Can you provide a quote from your essay?*
  
## Results

- Participants in each group had significantly different brain activity during essay writing
  - Brain only group had "stronger neural connectivity across all frequency bands" than LLM only group
  - The LLM only group had a "lower connectivity profile"
  - In other words, the brain only group had broader brain activity, as measured by the EEG
- The difference in information flowed differently
  - Brain only group had greater "bottom-up flows" (i.e. the brain generating novel ideas & figuring out how to convey them in writing)
  - LLM only group had greater "top-down flows" (i.e. reading and parsing and synthesizing the LLM output)
- The LLM group produced essays that were more similar to each other (as compared to the brain only group)
- The brain only group also had a greater ability to quote directly from the essay they had just written (although this went away in sessions #2 and #3, as participants knew the question was coming)

**Takeaway:** The paper argues that LLM use off-loads cognitive demand, but too much.  LLMs do the critical thinking for us, while we just synthesis and summarize.  This affects our ability to engage with new material and concepts in meaningful ways.  They make this claim via the EEG results, as well as the greater ability to quote (recall) what was just written for the brain only group.

# Thoughts & criticisms

- Thoughts
  - We should stay well informed on the effects of AI on education, including the good, the bad, and the ugly
- Critiques
  - The sample size is small & limited (54 college students)
  - I don't know enough neuroscience to fully understand the EEG results, but I doubt its a good signal for learning (I'd love to talk to someone who does know neuroscience though)
  - The participants had 20 minutes to write an essay response
    - That was it, nothing more
    - We should caution against making broad claims on human learning via 20 minute creative writing responses
  - The LLM only group just used vanilla ChatGPT.  No socratic method, no refined AI Assistant prompting to guide their responses.
  - The researchers (IMO) had non-SOTA AI knowledge and learning theory understanding
    - LLMs are described as "next token predictors"
      - I'd argue most AI researchers would not describe these systems like this, these systems do construct knowledge and world meaning in ways we don't fully understand
    - The paper mentions "learning styles" in the first paragraph of the introduction
      - Learning styles are widely agreed upon by educational researchers to be a myth (see [here](https://en.wikipedia.org/wiki/Learning_styles#Criticism), [here](https://fee.org/articles/learning-styles-don-t-actually-exist-studies-show/?gad_source=1&gad_campaignid=21607921915&gbraid=0AAAAADkIVmc-Xa2545ARjWZiX1AkEA3Dl&gclid=Cj0KCQjwwZDFBhCpARIsAB95qO2vQyEL0Hu43kGOOeAhYZy97IVbvEYV22HjyVUeKy16N9-Yx4ssVYMaAjmsEALw_wcB), [here](https://www.educationnext.org/stubborn-myth-learning-styles-state-teacher-license-prep-materials-debunked-theory/), and [here](https://onlineteaching.umich.edu/articles/the-myth-of-learning-styles/)) 
    - Frankly, the results aren't surprising (although interesting to see with the EEG technology)
      - Of course people will produce similar essays & have more top-down processing when giving only 20 minutes to work gpt-4o.  It's likely the same thing would happen if the person instead consulted an expert beforehand

# How this relates to building AI learning systems

- We already know that just giving students ChatGPT or an AI with no guardrails can harm content engagement and learning.   I'd like to see this same study repeated, but with a highly effective socratic AI tutor
- Many academic subjects do benefit from a combination of "top-down" and "bottom-up" thinking.  Math, for example.  Sure, I want a student to think creativity and solve problems in novel ways, but I also want them to understand how Pythagoras proved the theorem
