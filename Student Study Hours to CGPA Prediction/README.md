<a name="readme-top"></a>
# Student Study Hours to CGPA Prediction
### Predict academic performance utilizing polynomial regression and regularized ensembles.

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/ameythakur20/student-study-hours-to-cgpa-prediction)

---

# Hello fellow Kagglers!

This notebook presents a **robust regression framework** designed to predict student CGPA based on quantitative study metrics. The objective is to minimize the Mean Squared Error (MSE) by exploring the non-linear relationship between academic input (Study Hours) and performance output (CGPA).

Typical baseline approaches assume a constant rate of return on study time. However, this pipeline acknowledges that academic growth often follows a curvilinear path—exhibiting diminishing returns at higher intervals or accelerated learning phases. To address this, we implement a **polynomial expansion strategy** paired with **regularized learning benchmarks**.

The sections below walk through the mathematical formulation of our problem, the feature engineering logic, and the structural evaluation of our predictive ensemble.

---

## Understanding the Dataset

The competition provides a comprehensive dataset consisting of:
- **Study_Hours**: Total duration spent on academic activities.
- **CGPA**: Cumulative Grade Point Average (Target).

The dataset consists of 24,000 training records, providing a high-density environment for regression analysis and bias-variance tradeoff studies.

---

## Conceptual Modeling Strategy

Predicting academic outcomes is a **stochastic learning problem** where performance $\hat{y}$ is modeled as a function of temporal investment $x$.

We model this relationship via a **Polynomial Expansion**:

$$ \hat{y} = \beta_0 + \sum_{i=1}^{n} \beta_i x^i + \epsilon $$

### Key Workflow Stages

1. **Data Acquisition**: Loading train/test/submission files with staging validation.
2. **Exploratory Analytics**: Inspecting target distribution and bivariate correlations.
3. **Feature Engineering**: Implementing degree-2 polynomial expansion and standard scaling.
4. **Model Benchmarking**: Running a tournament between OLS, Ridge, Lasso, and Gradient Boosting.
5. **K-Fold Evaluation**: Ensuring prediction stability via 5-fold cross-validation.

---

## Technical Methodology

### 1. Polynomial Engineering
Standard linear regression fails to capture the subtle "curves" in student productivity. By creating an $x^2$ term, we allow the hypothesis function to bend, approximating the reality of academic plateauing or breakthroughs.

### 2. Regularization (L2)
With polynomial features, the model risks overfitting. We utilize **Ridge Regression** to add a penalty term to the cost function:
$$ J(\theta) = MSE + \alpha \sum \beta_i^2 $$
This forces the model to keep coefficients small, leading to better generalization on unseen student data.

---

## Summary of Results

The pipeline evaluates models based on **MSE (Mean Squared Error)** and **R² (Coefficient of Determination)**.
- **Top Performer**: Ridge Regression / Linear Regression (MSE: ~0.3642).
- **Inference Stability**: Demonstrated low standard deviation in cross-validation MSE, confirming model robustness.

---

## Closing Remarks

Forecasting educational outcomes requires a meticulous balance of statistical hygiene and model complexity. By transitioning from a simple linear baseline to an optimized **polynomial ensemble architecture**, we provide a scalable framework for academic diagnostics that is robust against local noise and global bias.

Discussion, experimentation, and alternative perspectives on the regression strategy are always welcome.

---

### Amey Thakur

[Kaggle](https://www.kaggle.com/ameythakur20) • [GitHub](https://github.com/Amey-Thakur)

---

<div align="center">

  [↑ Back to Top](#readme-top) &nbsp;·&nbsp; [← Back to Home](../README.md)

</div>
