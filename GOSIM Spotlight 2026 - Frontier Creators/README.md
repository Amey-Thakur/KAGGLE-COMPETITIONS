<a name="readme-top"></a>
# AI Hallucination Visualizer: Mapping Uncertainty in Generative Agents
### A diagnostic pipeline for quantifying and visualizing stochastic uncertainty in LLMs using GPT-2 token-level probability analysis.

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/architkonde/ai-hallucination-visualizer)

---

# Hello fellow Kagglers!

This notebook presents a **state-of-the-art diagnostic pipeline** initially showcased at *GOSIM Spotlight 2026: Frontier Creators*. The objective is to identify potential hallucination junctions in Large Language Models (LLMs) by visualizing the probabilistic reliability of generated text at the token level.

The solution integrates:

- **Logit-Space Extraction** during autoregressive generation
- **Confidence Discretization** into an actionable three-tier schema (High, Medium, Low)
- **Token Entropy Smoothing** via normalized transition scores
- **Dynamic HTML Visualization** mapping uncertainty via background alpha-channels
- **Path Competition Analysis** visualizing Top-K alternative candidates

Each component is designed to bridge the gap between black-box probabilistic outputs and **human-readable interpretability**.

---

## Understanding the Problem Setting

Detecting text hallucinations in LLMs is not just an "outcome-checking" exercise; it requires an understanding of **statistical certainty** and **high-entropy states**. Language models operate on probabilistic state transitions, where varying degrees of confidence dictate the next chosen token.

This task is modeled as a **transition probability problem**:

$$
P(w_t) = \exp(\text{transition\\_score}_t)
$$

### Key challenges

- intercepting raw scalar logits during standard causal language modeling
- normalizing generation scores to absolute probabilities $P(w_t | C)$
- presenting high-dimensional statistical competition as a digestible interface

---

## 1. Inference Pipeline & Score Extraction

The pipeline begins by engineering an **intervention in the generation loop**. By utilizing `output_scores=True` combined with `return_dict_in_generate=True`, we capture the internal state of the `GPT2LMHeadModel` without disrupting its forward pass, retrieving a multi-dimensional tensor of cumulative decision probabilities.

## 2. Mathematical Normalization

To translate unbounded logits into a strict probability space (0 to 1), we apply a normalization strategy utilizing `model.compute_transition_scores`.

This represents the normalized log-probability of the selected token relative to the vocabulary distribution at that specific timestep. By correcting the skewness of raw log-odds, we normalize the decision space, allowing us to accurately gauge certainty.

## 3. The Visual Discretization System

The core of the interface is a **three-tier heatmap architecture**. 

### Level 1: Factual Stability (High)
Green tokens (>60% confidence). These signify areas where the model is highly deterministic, often aligning with established factual or syntactic pathways.

### Level 2: Transitionary States (Medium)
Yellow tokens (30-60% confidence). These indicate stylistic variations where the model is choosing between highly probable synonyms.

### Level 3: Hallucination Junctions (Low)
Red tokens (<30% confidence). The "guessing" tier. High structural entropy often flags factual fabrications or knowledge boundaries.

---

## Summary

This pipeline demonstrates:

1. a professional approach to **Logit Extraction** and normalization
2. the massive performance gains of **Color-Coded Text Overlays**
3. the stability of an **uncertainty-aware inference** layer

This ensures that generated predictions are not only read, but their structural and factual integrity is statistically contextualized and resilient to the variance of the underlying sampling temperature.

---

## Closing Remark

Achieving structural transparency in foundation models requires a meticulous balance of tensor-level operations and UX design. By combining the precision of **PyTorch tensor analysis** with the visual accessibility of **HTML rendering**, we ensure that the model is entirely transparent in its "moment-of-doubt."

Further research will focus on:
- integrating sparse autoencoders to attribute low-confidence to missing feature activations
- evaluating cross-entropy thresholds across larger architectures (e.g., Llama-3, Gemma-2)
- analyzing token entropy distribution drift under adversarial prompts

---

### Amey Thakur

[Kaggle](https://www.kaggle.com/ameythakur20) • [GitHub](https://github.com/Amey-Thakur)

---

<div align="center">

  [↑ Back to Top](#readme-top) &nbsp;·&nbsp; [← Back to Home](../README.md)

</div>
