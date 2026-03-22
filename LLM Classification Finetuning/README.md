<a name="readme-top"></a>
# LLM Classification Finetuning

### Ensembled Pipeline Inference for Human Preference Classification

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/ameythakur20/llm-classification-inference)

---

## Overview

This repository features a state-of-the-art inference pipeline for the **LMSYS - Chatbot Arena Human Preference Predictions** competition. The objective is to predict which response a human would prefer when presented with two different outputs from large language models. The solution utilizes a high-capacity ensemble of fine-tuned Transformer models, optimized for stable execution on dual-GPU hardware.

## Strategic Methodology

The inference architecture is designed to maximize predictive accuracy while remaining within strict computational constraints:

1.  **Ensembled Scaling**: Combining the discriminative strengths of **Gemma-2-9B** (42 layers) and **Llama-3-8B** (32 layers) to capture diverse linguistic patterns.
2.  **Pipeline Parallelism**: Implementing a dual-GPU sharding strategy that splits layer execution across two NVIDIA Tesla T4 GPUs (15GB each) to prevent Out-Of-Memory (OOM) failures.
3.  **TTA (Test Time Augmentation)**: Counteracting positional bias by running inference on structurally swapped input pairs (Response A ↔ Response B) and geometrically remapping the resulting logit vectors.
4.  **Logit Interpolation**: Utilizing optimized weighting (57% Gemma, 43% Llama) to converge on final class probabilities.

---

### Amey Thakur

[Kaggle](https://www.kaggle.com/ameythakur20) • [GitHub](https://github.com/Amey-Thakur)

---

<div align="center">

  [↑ Back to Top](#readme-top) &nbsp;·&nbsp; [← Back to Home](../README.md)

</div>
