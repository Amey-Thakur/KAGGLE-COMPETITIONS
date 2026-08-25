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
The objective of this Playground Series competition is to predict the probability of smartphone addiction (`addicted_label`). The scoring metric is the Area Under the ROC Curve (AUC). As a low-dimensional tabular dataset of 691,369 training records, the primary constraint is capturing non-linear behavioural thresholds and synthetic generator artifacts without overfitting public test splits.

## What is here
| Stage | Description |
| :--- | :--- |
| **Exploration** | Identifies rank correlation, non-linear ratios, and distribution density across raw temporal features. |
| **Feature Engineering** | Constructs domain behavioural metrics (usage shares, notification intensity, group mean deviations), decimal lattice coordinates (`frac`, `d1`), and transductive frequency maps. |
| **Ensemble** | An Apex Multi-Seed Dual Meta-Stacking and Sovereign Logit-Stacking architecture combining Level-0 gradient-boosted probability signals and non-linear meta-learners. |

## Feature interaction strategy
Rather than introducing artificial missingness indicators that create distribution shift, missing values are handled through model-native mechanisms. To expose non-linear behavioural signals directly to the tree partitioners, domain interaction features are constructed:
1. `Non_Social_Screen_Time`: Screen time remaining after accounting for social media hours.
2. `Social_Share` & `Gaming_Share`: Proportion of screen time spent on high-dopamine activities.
3. `Notifications_per_Hour`: Frequency of notification interruptions normalised by screen duration.
4. `Group_Mean_Deviations`: Continuous residuals relative to demographic cohort baselines.
5. `Decimal_Lattice`: Sub-unit fractional offsets ($v - \lfloor v \rfloor$) and first-decimal digits capturing synthetic generator rounding patterns.

## Results
The validation scheme is a 5-Fold Stratified CV, preserving the exact class proportion of `addicted_label`. Level-0 out-of-fold predictions (`meta_lgb`, `meta_xgb`, `meta_cat`) along with interaction terms (`meta_prod`, `meta_diff`) are concatenated directly into the feature space to train Level-1 meta-learners.

| Model / Architecture | 5-Fold CV ROC AUC | Public LB Score |
| :--- | :---: | :---: |
| Single LightGBM Baseline | 0.96287 | 0.96182 |
| Multi-Seed Meta-Stacking | 0.96339 | 0.96509 |
| Apex Multi-Seed Dual Meta-Stacking | 0.96385 | **0.96525** |
| Sovereign Transductive Logit Stack | 0.9705+ | Pending |

## Where it fails
The stack is most confidently incorrect (predicting a high addiction probability for a negative label) when a user exhibits extreme social media usage alongside atypically low total screen time. The models require explicit non-linear interaction terms between total usage and activity shares to resolve edge cases.

## What would improve it
1. **Transductive Target & Frequency Encoding**: Jointly computing frequency distributions across combined train and test records to capture exact sample density manifolds.
2. **Native Ordered Statistics**: Providing raw string categorical representations directly to CatBoost to enable dynamic target-statistic trees.
3. **Logit-Space Stacking**: Training regularised meta-classifiers on logit transforms $\ln(p / (1-p))$ rather than raw probabilities to linearise boundary corrections.

---

<div align="center">

**Amey Thakur** &nbsp;·&nbsp; [Kaggle](https://www.kaggle.com/ameythakur20) &nbsp;·&nbsp; [GitHub](https://github.com/Amey-Thakur) &nbsp;·&nbsp; [ORCID](https://orcid.org/0000-0001-5644-1575)

<br>

[↑ Back to top](#predicting-smartphone-addiction) &nbsp;·&nbsp; [← Repository home](../../README.md)

</div>
