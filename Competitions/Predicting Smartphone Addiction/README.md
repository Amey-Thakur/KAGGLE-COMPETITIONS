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
| **Feature Engineering** | Constructs domain behavioural metrics (usage shares, notification intensity, group mean deviations) and rank normalisation. |
| **Ensemble** | A Multi-Stage Meta-Feature Stacking architecture using Level-0 out-of-fold probability signals and non-linear interactions to train a Level-1 meta-learner. |

## Feature interaction strategy
Rather than introducing artificial missingness indicators that create adversarial distribution shift, missing values are handled through model-native mechanisms. To expose non-linear behavioural signals directly to the tree partitioners, domain interaction features are constructed:
1. `Non_Social_Screen_Time`: Screen time remaining after accounting for social media hours.
2. `Social_Share` & `Gaming_Share`: Proportion of screen time spent on high-dopamine activities.
3. `Notifications_per_Hour`: Frequency of notification interruptions normalised by screen duration.
4. `Group_Mean_Deviations`: Continuous residuals relative to demographic cohort baselines.

## Results
The validation scheme is a 5-Fold Stratified CV, preserving the exact class proportion of `addicted_label`. Level-0 out-of-fold predictions (`meta_lgb`, `meta_xgb`, `meta_cat`) along with interaction terms (`meta_prod`, `meta_diff`) are concatenated directly into the feature space to train an XGBoost meta-learner, achieving an Out-Of-Fold CV AUC of **0.963556**.

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
