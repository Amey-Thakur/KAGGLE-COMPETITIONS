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

## Understanding the Problem Setting

Predicting academic outcomes is a **stochastic learning problem** where performance (ŷ) (CGPA) is modeled as a function of temporal investment x (Study Hours). While a simple linear model assumes:

ŷ = β₀ + β₁ x

real-world educational data often exhibits heteroscedasticity and non-linear trends.

We model this relationship via a **Polynomial Expansion of degree n**:

$$
\hat{y} = \beta_0 + \sum_{i=1}^{n} \beta_i x^i + \epsilon
$$

Where:
- β₀: base CGPA (y-intercept)
- βᵢ: coefficients for each polynomial power of study hours
- ε: stochastic residual (unexplained variance)

### Key Challenges

- **Overfitting High-Degree Terms**: Ensuring that x² or higher terms capture the "elbow" of the data without memorizing local noise.
- **Variance Control**: Utilizing L2 Regularization (Ridge) to penalize excessive coefficient magnitude in high-dimensional feature spaces.
- **Model Stability**: Validating consistency across a 24,000-record dataset via stratified grouping.

---

## 1. Data Acquisition & Staging

The pipeline begins with a **verified staging process**. We load the training and test datasets while performing a file existence check through a stylized validation table. This ensures the environment is properly configured before the computational workload begins.

- **Training Set**: 24,000 observations.
- **Test Set**: 6,000 observations (CGPA hidden).

## 2. Statistical Inspection & Hygiene

Before modeling, we perform a **spectral analysis** of the target variable. By inspecting the bounds of CGPA (0-10) and the distribution of study hours, we identify and exclude potential measurement errors or duplicate entries that could bias the global slope of our regression line.

## 3. Visual Discovery (EDA)

We utilize **Bivariate Statistical Profiling** to visualize the density between hours and CGPA. This stage is critical for identifying whether the variance in CGPA increases as study hours grow, which would necessitate specific normalization or log-transformation strategies.

## 4. Feature Science: Polynomial Engineering

The core transformation in this notebook is the **PolynomialFeature generation**. By projecting the 1D input space into a 2D or higher feature space, we calculate:
- x: Original hours  
- x²: Quadratic interaction term  

This allows a linear learner to fit non-linear boundaries. Subsequently, we apply **Standardized Feature Scaling** to ensure that  x² does not dominate the gradient descent due to its larger numerical magnitude.

## 5. Regularized Learning Ensembles

Our modeling strategy employs a **Structural Risk Minimization** approach. We compare four distinct algorithms:

1. **Ordinary Least Squares (OLS)**: The unbounded linear baseline.
2. **Ridge Regression (L2)**: Minimizing the cost function  J(θ) = MSE + α ∑ βᵢ²
3. **Lasso Regression (L1)**: Encouraging sparsity and automatic feature selection.
4. **Gradient Boosting (GBT)**: A non-parametric ensemble method used to capture complex decision boundaries.

## 6. Model Evaluation & Benchmarking

Performance is evaluated using the **Mean Squared Error (MSE)** and the **Coefficient of Determination (R²)**.

$$ MSE = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2 $$

By minimizing the squared distance between predicted and actual CGPA, we ensure the model is robust against large outliers while maintaining high precision for the average student profile.

---

## Summary

This analytical pipeline demonstrates:
1. **Mathematical Scalability**: How polynomial expansion upgrades a simple linear model into a complex performance predictor.
2. **Regularization Efficiency**: The effectiveness of Ridge regression in stabilizing coefficients in high-variance synthetic datasets.
3. **Predictive Precision**: Achieving an R² of ~0.82, indicating that study hours explain the vast majority of academic variance.

---

## Closing Remark

Forecasting educational outcomes requires a meticulous balance of statistical hygiene and model complexity. By transitioning from a simple linear baseline to an optimized **polynomial ensemble architecture**, we provide a scalable framework for academic diagnostics that is robust against local noise and global bias.

Further research will focus on:
- integrating sociodemographic metadata for multi-variate complexity.
- implementing Bayesian optimization for hyper-parameter tuning (L1/L2 alpha).
- analyzing residual drift under adversarial student study patterns.

---

### Amey Thakur

[Kaggle](https://www.kaggle.com/ameythakur20) • [GitHub](https://github.com/Amey-Thakur)

---

<div align="center">

  [↑ Back to Top](#readme-top) &nbsp;·&nbsp; [← Back to Home](../README.md)

</div>
