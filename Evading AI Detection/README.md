<a name="readme-top"></a>
# Evading AI-Generated Text Detection

### Activation Steering and Lexical Variance for Detection Evasion

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/ameythakur20/evading-ai-text-detection)

---

## Overview

This repository features a high-performance solution for the **Evading AI-Generated Text Detection** competition. The core strategy utilizes **activation steering** (forward hooks) to manipulate the internal latent states of a large language model (Gemma-2-2B) during generation. By suppressing specific AI-correlated features identified via Sparse Autoencoders (SAE), the model produces text that bypasses statistical discriminators while maintaining logical coherence.

## Methodology

The evasion pipeline operates across three primary dimensions:

1.  **Latent Feature Steering**: Suppressing internal activations related to formulaic structure and academic tone at Layer 12.
2.  **Entropy Maximization**: Utilizing high-temperature sampling (1.3) and top-k filtering to increase lexical diversity and unpredictability.
3.  **Lexical Intervention**: A post-processing pass that substitutes overused AI formalisms with neutral synonyms and probabilistically injects human-like contractions.

---

### Amey Thakur

[Kaggle](https://www.kaggle.com/ameythakur20) • [GitHub](https://github.com/Amey-Thakur)

---

<div align="center">

  [↑ Back to Top](#readme-top) &nbsp;·&nbsp; [← Back to Home](../README.md)

</div>
