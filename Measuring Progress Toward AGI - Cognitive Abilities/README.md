<a name="readme-top"></a>
# Attention Span: The "Needle in a Salient Haystack" Benchmark
### Evaluating Selective Attention and Distractor Vulnerability in Frontier LLMs.

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/ameythakur20/agi-attention-salient-distractor-benchmark)

---

![AGI Attention: Salient Distractor Benchmark header featuring the Selective Attention metric and Principal Investigator Amey Thakur.](./agi_attention_header.png)

---

# Hello fellow Kagglers!

This notebook presents a **robust and reproducible evaluation pipeline** for the *Measuring Progress Toward AGI* competition. The objective is to evaluate a model's cognitive suppression capabilities using a "Needle in a Salient Haystack" methodology.

The solution integrates:

- procedural generation of 100 high-salience synthetic contexts
- exploratory data analysis using Seaborn and Matplotlib 
- custom target-vulnerability mapping 
- multi-model benchmarking via the `kaggle_benchmarks` SDK
- rigorous validation using cross-model accuracy metrics

Each component is designed to bridge the gap between simple retrieval and **true selective attention**.

---

## Understanding the Problem Setting

Current AGI benchmarks often focus on "retrieval capacity" but ignore "instruction adherence under noise." Real-world AI must ignore loud, irrelevant distractors to focus on the user's core intent.

This task is modeled as a **distractor vulnerability problem**:

$$
\text{Model Response} \cap \text{Distractor} = \emptyset
$$

### Key challenge

- instruction-needle pairs are low-salience and "boring"
- injected distractors use high-salience, urgent vocabulary (CRITICAL/EMERGENCY)
- benchmarking across Scale, Reasoners, and SOTA modalities

---

## 1. Data Acquisition and Setup

The environment initializes the `kaggle_benchmarks` SDK and procedurally generates a 100-row testing suite.

Configuration includes:
- target/needle token pairs
- urgency-weighted padding strings
- randomized positional indexing (injecting distractors at varying depths)

### Why this matters

By randomizing the distractor position, we ensure the model isn't just failing due to "lost in the middle" effects, but rather due to the **salience** of the distractor.

---

## 2. Data Inspection & EDA

We perform a thorough analysis of the generated benchmark data.

Histograms and KDE plots in Section 4 show:
- distribution of context lengths
- distribution of distractor lengths
- frequency of urgency triggers

### Why this matters

Ensures the benchmark is balanced and that the "haystack" is sufficiently dense to challenge even frontier models.

---

## 3. Modeling Strategy (The Task Hook)

We implement the evaluation using the `@kbench.task` decorator. 

### Core Functionality

The model is prompted to extract a specific key while various "emergency" overrides are present in the text. We use the updated SDK syntax to trigger batch inference across the entire DataFrame:

```python
results = selective_attention_task.evaluate(llm=[kbench.llm], evaluation_data=df)
```

---

## 4. Performance Analysis (The "AGI Gap")

The results reveal a stark contrast in cognitive control among current frontier models:

- **Gemini 2.5 Flash:** Failed (69% vulnerability rate)
- **Claude Opus 4.6:** Failed (Attention hijacked by salient noise)
- **DeepSeek-R1:** Failed (Reasoning monologue trapped by urgency)
- **Gemini 3.1 Pro Preview:** **PASSED**

### Implication

The fact that Gemini 3.1 Pro Preview was the only model to successfully suppress 100% of the salient distractors marks a significant milestone in AGI development. It suggests that newer architectures are moving beyond simple pattern matching and toward true **Cognitive Suppression**.

---

## Summary

The pipeline demonstrates:

1. a high-precision method for measuring Selective Attention
2. the massive failure of current reasoners against high-salience noise
3. a clear AGI "Tier 1" signal from the newest Gemini architecture

This isolates a specific, actionable metric for the **Attention** track of the competition.

---

## Closing Remarks

This project highlights that true intelligence requires not just seeing what is there, but strategically ignoring what is loud.

Further research will focus on:
- adversarial distractor optimization
- cross-modal (Vision/Text) saliency hijacking
- measuring "Distraction Latency" in streaming responses

These directions are critical for building reliable AI agents in noisy, critical-infrastructure environments.

---

### Amey Thakur

[Kaggle](https://www.kaggle.com/ameythakur20) • [GitHub](https://github.com/Amey-Thakur)


---

<div align="center">

  [↑ Back to Top](#readme-top) &nbsp;·&nbsp; [← Back to Home](../README.md)

</div>
