<div align="center">

<!-- Medal block, only if one was earned. Above the title, image then caption. -->

# Predicting Smartphone Addiction

**Predict the probability of smartphone addiction based on usage and demographic features**

<br>

[![Author](https://img.shields.io/badge/Author-Amey_Thakur-0969DA)](https://www.kaggle.com/ameythakur20) [![ORCID](https://img.shields.io/badge/ORCID-0000--0001--5644--1575-A6CE39)](https://orcid.org/0000-0001-5644-1575) [![License](https://img.shields.io/badge/License-Apache_2.0-lightgrey)](https://www.apache.org/licenses/LICENSE-2.0)

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
The objective of this Playground Series competition is to predict the probability that a user is addicted to their smartphone (`addicted_label`). The scoring metric is Area Under the ROC Curve (AUC). The primary constraint here is correctly classifying users with non-linear behaviors—such as users with very high social media hours but overall low screen time—without relying on deep learning due to the tabular, low-dimensional nature of the dataset.

## What is here
| Stage | Description |
| :--- | :--- |
| **Exploration** | Identifies that no single continuous variable cleanly separates the classes below the extreme percentiles. |
| **Feature Engineering** | Constructs interaction ratios (e.g., social time relative to total screen time) to penalize high absolute values that lack context. |
| **Ensemble** | A blended model of XGBoost, LightGBM, and CatBoost to maximize AUC robustly across different categorical treatments. |

## Feature interaction strategy
The problem with raw hours of usage is that 2 hours of gaming out of 2 total screen hours is vastly different behavior than 2 hours of gaming out of 10. Rather than hoping the gradient boosters implicitly discover this division, explicit ratios (`Social_to_Screen_Ratio`, `Gaming_to_Screen_Ratio`) were engineered. This explicit encoding cost nothing in dimensionality but moved the local cross-validation AUC significantly by exposing the true density of usage.

## Results
The validation scheme used is a 5-Fold Stratified CV to preserve the exact class balance of `addicted_label` across folds. The unweighted mean blend of LightGBM, XGBoost, and CatBoost achieves an Out-Of-Fold (OOF) AUC that stably outperforms any single architecture, suggesting that the diverse categorical handling between CatBoost and LightGBM provides complimentary signal. 

## Where it fails
The blend is most confidently wrong (high probability prediction for a negative label) when a user has extreme social media usage but surprisingly low total screen time. The `Social_to_Screen_Ratio` captures the intensity, but the models lack a hard cutoff to discount users whose absolute time simply isn't long enough to qualify as addiction, regardless of the ratio. 

## What would improve it
1. **Absolute Time Thresholds**: Introducing a boolean feature for users with `< 2` hours total screen time to forcibly suppress high-probability scores from extreme ratios.
2. **Hyperparameter Tuning**: Running an Optuna study over LightGBM's `num_leaves` and `colsample_bytree` to refine the tree structures.
3. **Target Encoding**: Out-of-fold target encoding for the `Gender` categorical feature to provide a denser numeric signal before the first split.

---

<div align="center">

**Amey Thakur** &nbsp;·&nbsp; [Kaggle](https://www.kaggle.com/ameythakur20) &nbsp;·&nbsp; [GitHub](https://github.com/Amey-Thakur) &nbsp;·&nbsp; [ORCID](https://orcid.org/0000-0001-5644-1575)

<br>

[↑ Back to top](#predicting-smartphone-addiction) &nbsp;·&nbsp; [← Repository home](../../README.md)

</div>
