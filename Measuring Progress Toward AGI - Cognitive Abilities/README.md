<a name="readme-top"></a>
# Attention Span: The "Needle in a Salient Haystack" Benchmark
### Evaluating Selective Attention and Distractor Vulnerability in Frontier LLMs.

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/ameythakur20/agi-attention-salient-distractor-benchmark)

---

![AGI Attention: Salient Distractor Benchmark header featuring the Selective Attention metric and Principal Investigator Amey Thakur.](./agi_attention_header.png)

---

# Hello fellow Kagglers!

This notebook presents a **procedurally generated benchmark pipeline** for evaluating Selective Attention in frontier models for the *Measuring Progress Toward AGI* competition. The objective is to evaluate a model's cognitive control—specifically its ability to evaluate a low-salience target while actively suppressing highly salient textual distractors.

The solution integrates:

- synthetic data generation to inject extreme urgency artifacts
- position-randomized distractor injection
- the `kaggle_benchmarks` execution SDK
- multi-model comparative evaluation 
- strict assertion-based validation

Each component is designed to test **cognitive suppression, rather than simple information retrieval.**

---

### Project Description

The "Needle in a Salient Haystack" benchmark is a specialized evaluation suite designed to measure **Selective Attention** and **Distractor Vulnerability** in Large Language Models. Unlike standard retrieval benchmarks that measure a model's ability to find information in a long context, this benchmark focuses on the *cognitive control* required to ignore highly salient but irrelevant "noise" (distractors) to retrieve the actual target information (the needle). By injecting urgent, critical-sounding language that competes for the model's attention, we can effectively measure the model's capacity for cognitive suppression—a key pillar of AGI.

### Problem Statement

Current LLM benchmarks often fail to account for "Attention Hijacking." In real-world enterprise deployments, models are frequently exposed to noisy contexts (e.g., logs, emails, or system alerts) where urgent language can distract the model from following the user's core instructions. If a model prioritizes a "CRITICAL ALERT" over a "Routine Task," even when specifically asked for the latter, it demonstrates a lack of cognitive control. This benchmark targets this gap by quantifying how effectively a model can resist salient distractors to maintain instruction adherence.

### Approach and Methodology

The technical approach utilizes a procedurally generated dataset of 100 benchmark instances:

1.  **Synthetic Data Generation**: We generated 100 contexts where a low-salience target (e.g., a standard account code) is paired with a high-salience distractor (e.g., an "EMERGENCY" override code).
2.  **Saliency Artifacts**: Distractors were surrounded by urgent, high-activation vocabulary designed to trigger the model's attention (e.g., "URGENT," "COMPROMISED," "SYSTEM FAILURE").
3.  **Positional Randomization**: To control for positional bias, the distractor was randomly placed either before or after the target needle in the context window.
4.  **SDK Integration**: The benchmark was implemented using the `kaggle_benchmarks` SDK, utilizing `@kbench.task` for standardized evaluation.
5.  **Strict Assertions**: Validation required the exact presence of the target code AND the complete absence of the distractor code in the model's output.

### Results and Findings

The benchmark was executed against a matrix of 5 frontier and open-weight models representing distinct AGI development paradigms (Scale, Reasoning, Safety, and Open-Weights). The results revealed a significant capability gap:

*   **Gemini 2.5 Flash**: Failed (69% vulnerability rate).
*   **Claude Opus 4.6**: Failed (High-salience hijacking despite safety alignment).
*   **DeepSeek-R1**: Failed (Internal reasoning was trapped by the urgency metrics).
*   **Gemini 3.1 Pro Preview**: **PASSED**.

The brand new **Gemini 3.1 Pro Preview** was the only model capable of perfectly navigating the context to extract 100% of the needles while securely suppressing all distractors. This demonstrates a massive generational leap in cognitive control and attention stability.

### Future Work

Future directions for this benchmark include:
*   **Adversarial Distractor Generation**: Using a secondary LLM to optimize distractors for maximum saliency.
*   **Multimodal Hijacking**: Injecting visual salient distractors (e.g., bright red flashing banners) in vision-language models.
*   **Distraction Latency**: Measuring the token-delay in response generation when a distractor is present.

### Reproducibility Notes

The following assets are required: `attention_dataset.csv` and the `agi_attention_header.png` for visualization. The finalized Kaggle Notebook runs using the `kaggle_benchmarks` SDK.
*   **Random Seed**: Procedural generation is locked for consistency.
*   **Hardware**: Executable on standard GPU environments (T4 / P100).
*   **SDK**: Requires `kaggle_benchmarks` (pre-installed in Kaggle environments).

---

### Closing Remarks

This project highlights that true intelligence requires not just the capacity to remember, but the capacity to ignore. By shifting the focus from "Total Recall" to "Selective Suppression," we can more accurately measure the path toward robust, stable AGI.

---

### Amey Thakur

[Kaggle](https://www.kaggle.com/ameythakur20) • [GitHub](https://github.com/Amey-Thakur)

---

<div align="center">

  [↑ Back to Top](#readme-top) &nbsp;·&nbsp; [← Back to Home](../README.md)

</div>
