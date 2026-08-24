<div align="center">

<!-- Medal block, only if one was earned. Above the title, image then caption. -->

# Predicting Smartphone Addiction

**Predict the probability of smartphone addiction based on usage and demographic features**

<br>

[![Notebook](https://img.shields.io/badge/Notebook-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/ameythakur20/predicting-smartphone-addiction) [![Author](https://img.shields.io/badge/Author-Amey_Thakur-0969DA)](https://www.kaggle.com/ameythakur20) [![ORCID](https://img.shields.io/badge/ORCID-0000--0001--5644--1575-A6CE39)](https://orcid.org/0000-0001-5644-1575) [![License](https://img.shields.io/badge/License-Apache_2.0-lightgrey)](https://www.apache.org/licenses/LICENSE-2.0)

<br>

**Notebook in this folder**

[`predicting-smartphone-addiction.ipynb`](./predicting-smartphone-addiction.ipynb)

<br>

<a href="https://www.kaggle.com/code/ameythakur20/predicting-smartphone-addiction"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open in Kaggle"></a>

<br>

[Competitions](../../README.md#competitions) &nbsp;·&nbsp; [Achievements](../../Achievements/Badges/README.md) &nbsp;·&nbsp; [Courses](../../Kaggle%20Courses/README.md) &nbsp;·&nbsp; [Kaggle Profile](https://www.kaggle.com/ameythakur20)

</div>

---

## The problem
The objective of this Playground Series competition is to predict the probability of smartphone addiction (`addicted_label`). The scoring metric is the Area Under the ROC Curve (AUC). As a low-dimensional tabular dataset, the primary constraint is correctly classifying users with non-linear behavioural thresholds without relying on parameter-heavy deep learning architectures.

## What is here
| Stage | Description |
| :--- | :--- |
| **Exploration** | Identifies rank correlation and distribution density across raw temporal features. |
| **Feature Engineering** | Constructs rank-transformed features to normalise skewness and explicit boolean flags for missing values to capture structural absence. |
| **Ensemble** | A Level-1 Stacking architecture using an optimal constrained SLSQP blend over the out-of-fold predictions of XGBoost, LightGBM, and CatBoost. |

## Feature interaction strategy
Gradient boosting architectures natively handle dense tabular data, yet they remain vulnerable to extreme outliers during tree splitting. Applying a rank transformation to highly skewed continuous variables (such as `Notifications_Count`) provides a uniformly distributed ordinal signal. This explicitly stabilises the ensemble without incurring dimensionality costs, improving the Out-Of-Fold (OOF) AUC.

## Results
The validation scheme is a 5-Fold Stratified CV, preserving the exact class proportion of `addicted_label`. A constrained SLSQP metric-optimiser dynamically weights the OOF predictions of the three base models, ensuring non-negative contributions that strictly maximise the ROC AUC.

## Where it fails
The stack is most confidently incorrect (predicting a high addiction probability for a negative label) when a user exhibits extreme social media usage alongside atypically low total screen time. The models lack a strict conditional cut-off to discount users whose absolute usage time is insufficient to mathematically qualify as addiction.

## What would improve it
1. **Absolute Time Thresholds**: Introducing a Boolean feature for users with fewer than 2 hours of total screen time to suppress high-probability scores mechanically.
2. **Hyperparameter Tuning**: Executing an Optuna study across LightGBM's `num_leaves` and `colsample_bytree` parameters to optimise tree structures.
3. **Target Encoding**: Out-of-fold target encoding for the `Gender` categorical feature to provide a denser numerical signal prior to the root split.

---

<div align="center">

**Amey Thakur** &nbsp;·&nbsp; [Kaggle](https://www.kaggle.com/ameythakur20) &nbsp;·&nbsp; [GitHub](https://github.com/Amey-Thakur) &nbsp;·&nbsp; [ORCID](https://orcid.org/0000-0001-5644-1575)

<br>

[↑ Back to top](#predicting-smartphone-addiction) &nbsp;·&nbsp; [← Repository home](../../README.md)

</div>
