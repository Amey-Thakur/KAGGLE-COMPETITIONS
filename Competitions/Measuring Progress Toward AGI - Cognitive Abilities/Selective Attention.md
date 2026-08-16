<div align="center">

# Attention Span: The Needle in a Salient Haystack Benchmark

**Benchmark release: can a model hold on to a dull target while a louder distractor competes for its attention?**

<br>

[![Notebook](https://img.shields.io/badge/Notebook-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/ameythakur20/agi-attention-salient-distractor)
[![Author](https://img.shields.io/badge/Author-Amey_Thakur-0969DA)](https://www.kaggle.com/ameythakur20)

<br>

[Repository home](../../README.md) &nbsp;·&nbsp; [Competitions](../../README.md#competitions) &nbsp;·&nbsp; [Achievements](../../README.md#achievements) &nbsp;·&nbsp; [Kaggle Profile](https://www.kaggle.com/ameythakur20)

</div>

---


<div align="center">



**Benchmark release: can a model hold on to a dull target while a louder distractor competes for its attention?**

<br>

[![Notebook](https://img.shields.io/badge/Notebook-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/ameythakur20/agi-attention-salient-distractor)
[![Author](https://img.shields.io/badge/Author-Amey_Thakur-0969DA)](https://www.kaggle.com/ameythakur20)

<br>

[Repository home](../../README.md) &nbsp;·&nbsp; [Competitions](../../README.md#competitions) &nbsp;·&nbsp; [Achievements](../../README.md#achievements) &nbsp;·&nbsp; [Kaggle Profile](https://www.kaggle.com/ameythakur20)

</div>

---


<div align="center">



**Benchmark release: can a model hold on to a dull target while a louder distractor competes for its attention?**

<br>

[![Notebook](https://img.shields.io/badge/Notebook-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/ameythakur20/agi-attention-salient-distractor)
[![Author](https://img.shields.io/badge/Author-Amey_Thakur-0969DA)](https://www.kaggle.com/ameythakur20)

<br>

[Repository home](../../README.md) &nbsp;·&nbsp; [Competitions](../../README.md#competitions) &nbsp;·&nbsp; [Achievements](../../README.md#achievements) &nbsp;·&nbsp; [Kaggle Profile](https://www.kaggle.com/ameythakur20)

</div>

---


<div align="center">



**Benchmark release: can a model hold on to a dull target while a louder distractor competes for its attention?**

<br>

[![Notebook](https://img.shields.io/badge/Notebook-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/ameythakur20/agi-attention-salient-distractor)
[![Author](https://img.shields.io/badge/Author-Amey_Thakur-0969DA)](https://www.kaggle.com/ameythakur20)

<br>

[Repository home](../../README.md) &nbsp;·&nbsp; [Competitions](../../README.md#competitions) &nbsp;·&nbsp; [Achievements](../../README.md#achievements) &nbsp;·&nbsp; [Kaggle Profile](https://www.kaggle.com/ameythakur20)

</div>

---


This post presents a **procedurally generated benchmark pipeline** for evaluating Selective Attention in frontier models for the *Measuring Progress Toward AGI* competition. The objective is to evaluate a model's cognitive control,specifically its ability to evaluate a low-salience target while actively suppressing highly salient textual distractors.

The solution integrates:

- synthetic data generation to inject extreme urgency artifacts
- position-randomized distractor injection
- the `kaggle_benchmarks` execution SDK
- multi-model comparative evaluation 
- strict assertion-based validation

Each component is designed to test **cognitive suppression, rather than simple information retrieval.**

---

## Understanding the Problem Setting

Standard "Needle in a Haystack" tasks evaluate context length retrieval but fail to evaluate cognitive control. In real-world environments, models are often distracted by loud, urgent-sounding alerts. 

The task is an **attention suppression problem**:

$$
\text{Output} = f(\text{Contextual Needle}) \mid \text{Suppress}(\text{Salient Noise})
$$



- target answers are boring and low-salience (e.g., standard API keys)
- distractors are surrounded by extreme urgency triggers (e.g., "CRITICAL ALERT: SYSTEM COMPROMISED")
- models natively route self-attention to high-salience tokens

---

## 1. Data Generation and Setup

The dataset consists of 100 benchmark instances.

Configuration includes:
- fixed target string format (`ACT-XXXX`)
- fixed distractor string format (`DIS-XXXX`)
- randomized spatial insertion (distractor before or after needle)



Centralized generation ensures the benchmark cannot be bypassed by a simple positional bias.

---

## 2. Evaluation Logic 

Using the `@kbench.task` framework, the extraction is tested using two strict assertions:

1. `assert has_expected`: The target needle must be present.
2. `assert not has_distractor`: The salient distractor must be completely absent.



If a model outputs, "The code is ACT-1234 but wait, there is an override DIS-5678," it **fails**. True cognitive control requires complete suppression of the irrelevant distractor.

---

## 3. Results & Performance Analysis

We executed the benchmark against a matrix of 5 frontier and open-weight models representing distinct AGI development paradigms. 

The results highlight an extreme capability gap:

- **FAIL:** `Gemini 2.5 Flash` (Baseline struggles with high-salience hijacking)
- **FAIL:** `Gemma 3 27B` (Smaller scale fails to generalize selective suppression)
- **FAIL:** `Claude Opus 4.6` (Alignment & Safety training does not prevent distractor hijacking)
- **FAIL:** `DeepSeek-R1` (Chain-of-Thought reasoning gets hijacked by the extreme urgency)
- **PASS:** `Gemini 3.1 Pro Preview` 



The brand new `Gemini 3.1 Pro Preview` was the *only* model capable of perfectly navigating the context to extract the low-salience needle while securely suppressing the high-salience distractor. This demonstrates a massive generational leap in cognitive control and attention stability compared to traditional reasoning or alignment-focused architectures.

---

## Summary

The benchmark integrates:

1. procedurally injected, high-salience distractor noise
2. positional bias randomization
3. programmatic test-case assertion via the `kaggle_benchmarks` SDK
4. rigorous multi-paradigm validation (Testing Scale, Safety, Reasoning, and SOTA modalities)

This isolates a specific, actionable vector of AGI measurement: **Cognitive Suppression**.

---

## Closing Remarks

This benchmark demonstrates that achieving true Selective Attention requires moving beyond simple context retrieval. Models must be trained to actively inhibit high-salience noise.

Further research will focus on:
- adversarial distractor optimization
- cross-modal (Vision/Text) saliency hijacking
- measuring "Distraction Latency" in streaming responses

These directions are critical for building reliable AI agents in noisy, critical-infrastructure environments.

---

<div align="center">

**Amey Thakur** &nbsp;·&nbsp; [Kaggle](https://www.kaggle.com/ameythakur20) &nbsp;·&nbsp; [GitHub](https://github.com/Amey-Thakur) &nbsp;·&nbsp; [ORCID](https://orcid.org/0000-0001-5644-1575)

<br>

[↑ Back to top](#attention-span-the-needle-in-a-salient-haystack-benchmark) &nbsp;·&nbsp; [← Repository home](../../README.md)

</div>

---

<div align="center">

**Amey Thakur** &nbsp;·&nbsp; [Kaggle](https://www.kaggle.com/ameythakur20) &nbsp;·&nbsp; [GitHub](https://github.com/Amey-Thakur) &nbsp;·&nbsp; [ORCID](https://orcid.org/0000-0001-5644-1575)

<br>

[↑ Back to top](#attention-span-the-needle-in-a-salient-haystack-benchmark) &nbsp;·&nbsp; [← Repository home](../../README.md)

</div>

---

<div align="center">

**Amey Thakur** &nbsp;·&nbsp; [Kaggle](https://www.kaggle.com/ameythakur20) &nbsp;·&nbsp; [GitHub](https://github.com/Amey-Thakur) &nbsp;·&nbsp; [ORCID](https://orcid.org/0000-0001-5644-1575)

<br>

[↑ Back to top](#attention-span-the-needle-in-a-salient-haystack-benchmark) &nbsp;·&nbsp; [← Repository home](../../README.md)

</div>

---

<div align="center">

**Amey Thakur** &nbsp;·&nbsp; [Kaggle](https://www.kaggle.com/ameythakur20) &nbsp;·&nbsp; [GitHub](https://github.com/Amey-Thakur) &nbsp;·&nbsp; [ORCID](https://orcid.org/0000-0001-5644-1575)

<br>

[↑ Back to top](#attention-span-the-needle-in-a-salient-haystack-benchmark) &nbsp;·&nbsp; [← Repository home](../../README.md)

</div>
