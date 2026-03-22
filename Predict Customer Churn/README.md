<a name="readme-top"></a>
# Predict Customer Churn <img align="right" src="../Medals/Bronze Medal.png" width="35" title="Bronze Medal (Mar 20, 2026)">
### Playground Series - Season 6 Episode 3

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/ameythakur20/predict-customer-churn-xgb-catboost-lgbm-optuna)

---

# Hello fellow Kagglers!

This notebook presents a customer churn prediction pipeline built using an ensemble of gradient boosting models: **XGBoost**, **LightGBM**, and **CatBoost**, with hyperparameter tuning performed using **Optuna**.

The purpose of this explanation is to clarify how the pipeline is structured, why each component is used, and how the individual models contribute to the final prediction.

Rather than focusing only on model training, the discussion follows the actual implementation in the notebook, covering data preparation, validation, optimization, and ensembling in the same order.

---

## Data Preparation

The dataset is separated into:

* feature matrix `X`
* target variable `y`

Each row represents a customer, and the target variable is defined as:

* `1` indicates churn
* `0` indicates retention

The model estimates the probability:

$$
P(y = 1 \mid x)
$$

which represents the likelihood of a customer leaving the service.

In this notebook, preprocessing is intentionally minimal. Missing values are handled by filling numerical features with a statistical value and assigning a default category to categorical features. This ensures consistent input across all models without introducing unnecessary transformations.

---

## Handling Categorical Features

Categorical variables are processed differently depending on the model used in the notebook.

For **XGBoost** and **LightGBM**, categorical features are converted into numeric form before training. This allows the models to perform standard tree-based splits.

For **CatBoost**, categorical feature indices are passed directly without manual encoding. The model internally computes statistics of the form:

$$
\mathbb{E}[y \mid \text{category}]
$$

This approach preserves category-level information and reduces the need for preprocessing.

---

## Validation Strategy

The notebook uses a train-validation split to evaluate model performance during training and tuning.

Model quality is measured on validation data rather than training data to ensure that performance reflects generalization. When training performance is significantly better than validation performance, the model is overfitting.

This validation setup is also used consistently during hyperparameter optimization.

---

## Model Training Objective

All models in the notebook optimize log loss, defined as:

$$
\mathcal{L} =
-\frac{1}{n} \sum
\left[
y \log(p) + (1-y)\log(1-p)
\right]
$$

This loss function penalizes incorrect predictions, especially when the model is confident but wrong.

For example, predicting 0.2 when the true label is 1 results in a larger penalty than predicting 0.8 for the same case. This makes log loss appropriate for probability estimation in churn prediction.

---

## Gradient Boosting Framework

The models follow the gradient boosting approach implemented in their respective libraries.

Predictions are constructed incrementally as a sum of decision trees:

$$
F(x) = f_1(x) + f_2(x) + \dots + f_T(x)
$$

During training, each iteration fits a new tree to correct the errors of the current model. In practical terms, the correction signal can be written as:

$$
\text{error} = y - p
$$

If predictions are too low, subsequent trees increase them. If predictions are too high, subsequent trees reduce them. This process continues until the model converges or reaches the defined number of iterations.

---

## Model Differences

Although all models use boosting, their implementations differ in ways that affect performance.

* **XGBoost** grows trees level by level and applies regularization, resulting in stable and controlled learning behavior.
* **LightGBM** grows trees leaf-wise and prioritizes splits with the largest error reduction, which accelerates learning but increases the risk of overfitting.
* **CatBoost** uses ordered boosting and handles categorical features directly, reducing target leakage and improving performance on categorical data.

These differences lead to complementary behavior across models.

---

## Hyperparameter Optimization

Hyperparameter tuning is performed using Optuna within the notebook.

For each trial, a set of parameters is sampled, the model is trained using the training split, and performance is evaluated on the validation split. The objective is to minimize validation loss:

$$
\theta^* = \arg\min \mathcal{L}_{val}
$$

The search process focuses on parameter regions that produce better validation performance, allowing the notebook to identify effective configurations without manual tuning.

---

## Model Predictions

After training, each model produces probability estimates for the target:

* XGBoost → $p_{\text{xgb}}(x)$
* LightGBM → $p_{\text{lgb}}(x)$
* CatBoost → $p_{\text{cat}}(x)$

These predictions are generated using the trained models on the same feature set.

---

## Ensembling Strategy

The notebook combines predictions using simple averaging:

$$
\hat{p}(x) =
\frac{
p_{\text{xgb}} + p_{\text{lgb}} + p_{\text{cat}}
}{3}
$$

This choice is intentional. Averaging reduces variance and improves stability without introducing additional complexity. Because the models differ in how they learn patterns, their errors are not identical, and averaging produces more reliable predictions.

---

## Final Prediction

The final classification is obtained by applying a threshold to the averaged probability:

$$
\hat{y} =
\begin{cases}
1 & \hat{p}(x) > 0.5 \\
0 & \text{otherwise}
\end{cases}
$$

This converts probability estimates into binary churn predictions required for submission.

---

## Computational Considerations

The most computationally intensive component of the notebook is hyperparameter optimization.

Training time depends on:

* the number of Optuna trials
* the number of boosting iterations
* the dataset size

Early stopping is used during training to prevent unnecessary computation and reduce overfitting.

---

## Closing Remarks

This notebook demonstrates a structured approach to customer churn prediction using gradient boosting and model ensembling.

By combining multiple models, applying systematic hyperparameter tuning, and using a simple averaging strategy, the pipeline achieves stable and consistent performance on tabular data.

Further improvements can be explored through feature engineering, alternative ensemble methods, or additional model variants.

---

### Amey Thakur

[Kaggle](https://www.kaggle.com/ameythakur20) • [GitHub](https://github.com/Amey-Thakur)

---

<div align="center">

  [↑ Back to Top](#readme-top) &nbsp;·&nbsp; [← Back to Home](../README.md)

</div>
