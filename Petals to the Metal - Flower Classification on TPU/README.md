<a name="readme-top"></a>
# Petals to the Metal - Flower Classification on TPU
### Getting Started with TPUs on Kaggle!

<!-- [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](Notebook Link Coming Soon) -->

---

# Hello fellow Kagglers!

This repository contains an advanced TPU-based flower classification architecture. The primary goal is to maximize the macro F1 score across 104 target classes using a triple-backbone ensemble.

### Notebooks

- **[Triple-Backbone Ensemble (Score Potential: 0.99+)](tpu-flower-classification-triple-ensemble.ipynb)**: A probability-averaged ensemble of EfficientNetB7, DenseNet201, and Xception with CutMix augmentation and label smoothing.
- **[Advanced Ensemble (Score: 0.94192)](tpu-flower-classification-advanced-ensemble.ipynb)**: Initial dual-stream architecture (EfficientNetB6 + DenseNet201).

### Architectural Overview

1. **Diverse Backbones**: Independent training of EfficientNetB7, DenseNet201, and Xception to ensure gradient diversity.
2. **Stochastic Regularization**: Integration of CutMix augmentation to force the network to learn holistic features.
3. **Smooth Targets**: Implementation of label smoothing (0.1) to improve generalization on visually similar classes.
4. **Ensemble TTA**: A 5-step Test Time Augmentation (TTA) pipeline combined with probability averaging across all architectures.

---

### Amey Thakur

[Kaggle](https://www.kaggle.com/ameythakur20) • [GitHub](https://github.com/Amey-Thakur)

---

<div align="center">

  [↑ Back to Top](#readme-top) &nbsp;·&nbsp; [← Back to Home](../README.md)

</div>
