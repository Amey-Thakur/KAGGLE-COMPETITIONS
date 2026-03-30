<a name="readme-top"></a>
# CS Week Codeathon AIML (Easy Level): Student Final Score Prediction
### A predictive modeling approach using Exploratory Data Analysis (EDA) and Feature Engineering (FE) to forecast academic performance.

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/ameythakur20/student-final-score-prediction-with-eda-and-fe)

---

# Hello fellow Kagglers!

This notebook presents a student final score prediction pipeline built around **exploratory data analysis**, **target-focused feature engineering**, and **MAE-based regression comparison**.

The purpose of this explanation is to clarify why each stage appears in the notebook, how the modeling decisions follow from the observed data patterns, and where further improvements can still be explored.

Rather than describing the notebook only at a high level, the discussion follows the actual implementation order: data reading, inspection, exploratory analysis, engineered variables, regression benchmarking, residual analysis, and final ensembling.

---

## Problem Framing

The competition asks for prediction of a continuous target:

* `final_score`

using student-level academic and lifestyle variables:

* `hours_studied`
* `attendance`
* `previous_score`
* `sleep_hours`
* `assignments_completed`

This is a supervised regression task. In notation, the objective is to estimate a function

$$
\hat{y} = f(x)
$$

where:

* `x` is the feature vector for one student
* `ŷ` is the predicted final exam score

The evaluation metric is **Mean Absolute Error**:

$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

This metric is useful here because it measures prediction error directly in score units. A lower value means predicted scores are closer to the true scores on average.

---

## Why The Notebook Starts With Direct Data Inspection

The notebook begins with shape checks, head displays, data types, and summary statistics.

This is done for two reasons:

1. to confirm that the train and test files have the expected structure
2. to understand whether the dataset requires heavy preprocessing before modeling

The inspection step shows that:

* all predictors are numeric
* the dataset is small
* train and test distributions are reasonably similar
* missing values are absent
* duplicate rows are absent

That matters because a small, clean, fully numeric dataset often responds well to relatively simple models. In such settings, excessive preprocessing can add unnecessary noise rather than useful signal.

---

## Why Exploratory Data Analysis Is Important Here

The EDA section is not included only for presentation. It directly informs the modeling choices that follow.

### 1. Target Distribution

The histogram and boxplot for `final_score` help inspect:

* score range
* central tendency
* spread
* possible outliers

This is useful because regression models behave differently when the target is highly skewed, extremely bounded, or dominated by a few large outliers.

### 2. Train-Test Distribution Comparison

The notebook compares train and test distributions feature by feature.

This step matters because leaderboard performance depends not only on fitting the training set, but also on how well the learned structure transfers to the unseen test data. If test distributions differ too much, a model with strong in-sample fit can still underperform at submission time.

### 3. Feature-Target Scatter and Trend Plots

The scatter plots and regression lines help answer practical questions such as:

* Is the relationship mostly linear?
* Are there visible nonlinear bends?
* Are interactions likely?
* Does any feature appear weak or unstable?

These plots are especially important in a small tabular problem. With only a few original variables, the visual relationship between each predictor and the target often gives direct clues about useful engineered terms.

### 4. Correlation Analysis

The correlation matrix and ranked target-correlation table help identify the strongest first-order relationships.

In simple form, the notebook is inspecting values like:

$$
\mathrm{corr}(x_j, y)
$$

for each feature `x_j`.

A strong correlation does not prove causality, but it helps identify where the signal is concentrated. This is one reason the notebook keeps the feature design centered on the original academic indicators rather than constructing a very large synthetic feature space immediately.

### 5. Quantile-Based Mean Plots

The quantile plots help study how the average target changes across ordered bins of each feature.

This is useful because raw scatter plots can look noisy. Quantile averages provide a smoother view of whether the response is:

* increasing
* decreasing
* saturating
* slightly curved

That information motivates the inclusion of a few squared terms and interaction terms later in the notebook.

---

## Why The Feature Engineering Is Structured This Way

The engineered features in this notebook are not arbitrary additions. They are designed to test specific hypotheses about how student performance may be formed.

---

## Interaction Features

The notebook creates interaction features such as:

* `hours_x_attendance`
* `hours_x_previous`
* `attendance_x_previous`
* `assignments_x_attendance`
* `sleep_x_hours`

The reason for adding these is that academic performance is rarely determined by each variable in isolation.

For example:

* the value of studying hours may depend on prior academic strength
* attendance may matter more when prior performance is already high
* assignment completion may carry more signal when attendance is also strong

Mathematically, these are terms of the form:

$$
x_i x_j
$$

which allow the model to learn joint effects instead of only separate additive effects.

A purely linear model estimates:

$$
\hat{y} = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_px_p
$$

After interaction terms are added, the model can also estimate effects like:

$$
\hat{y} = \beta_0 + \sum_{j} \beta_j x_j + \sum_{i < j} \beta_{i,j} x_i x_j
$$

This is often useful in small, clean numeric datasets where the hidden relationship is not fully independent across variables.

---

## Ratio Features

The notebook includes variables such as:

* `hours_per_assignment`
* `assignments_per_hour`
* `previous_per_sleep`
* `attendance_per_sleep`

The motivation here is that raw values do not always tell the full story. Sometimes relative balance matters more.

For example:

* many study hours with very few assignments completed can indicate inefficiency
* a high previous score relative to low sleep may indicate a different performance profile than the same score with healthy sleep
* attendance relative to sleep may capture consistency or strain

These features are an attempt to capture proportional relationships that simple additive models cannot see directly.

---

## Difference Features

The notebook also includes:

* `previous_minus_attendance`
* `attendance_minus_sleep`
* `hours_minus_sleep`

These are added to measure contrast between related variables.

Difference-based features can be useful when the gap between two signals carries information beyond their individual magnitudes. For instance, a large mismatch between previous score and attendance may reflect instability in academic pattern.

---

## Quadratic Features

The squared terms in the notebook are:

* `hours_studied_sq`
* `attendance_sq`
* `previous_score_sq`
* `sleep_hours_sq`
* `assignments_completed_sq`

These are included to allow mild curvature in the fitted relationship.

A linear model assumes one unit of change in a feature has a constant effect everywhere. In practice, some features may show diminishing returns or mild nonlinear response. A squared term adds this flexibility through expressions like:

$$
\hat{y} = \beta_0 + \beta_1x + \beta_2x^2
$$

This is especially reasonable for variables like `sleep_hours`, where both too little and too much may be less favorable than a balanced range.

---

## Aggregate Features

The notebook adds broader summary variables such as:

* `academic_input_sum`
* `academic_balance_mean`
* `academic_balance_std`

These are intended to compress several related inputs into a single summary of academic intensity or internal variability.

The reason for testing these features is that a student's overall profile may sometimes be easier to model through a compact summary than through all raw variables separately.

---

## Why Multiple Model Families Are Compared

The notebook does not assume one model family is best. Instead, it compares several approaches under the same validation setup:

* Linear Regression
* Ridge Regression
* Elastic Net
* Polynomial Ridge
* Random Forest
* Extra Trees
* HistGradientBoosting
* KNN
* CatBoost
* LightGBM
* XGBoost

This comparison serves two purposes:

1. it identifies which type of model best matches the structure of the dataset
2. it prevents overcommitting early to a single modeling assumption

For a competition like this, that matters because the target may be:

* mostly additive
* mildly nonlinear
* interaction-driven
* threshold-driven
* partially synthetic with hidden generation structure

The benchmark section is therefore a model-diagnostic step, not just a leaderboard step.

---

## Why The Validation Setup Uses OOF MAE

The notebook uses 10-fold KFold cross-validation with shuffling.

For each model, out-of-fold predictions are collected and then scored with MAE.

This is important because out-of-fold predictions approximate how the model behaves on unseen data. Training error alone would be misleading, especially for more flexible models.

The OOF framework also makes model comparisons fair, since every method is evaluated under the same split structure.

---

## Why Linear Models Perform Best In This Notebook

One of the most important findings in the notebook is that the strongest results come from:

* `linear_regression`
* `ridge_regression`
* `elastic_net`

while tree models and full polynomial expansion underperform.

This suggests that the main signal in the competition data is likely:

* smooth
* additive or near-additive
* driven by a few structured interactions
* not heavily threshold-based

That is an important modeling conclusion.

A strong tree model usually becomes more useful when the data contains many irregular splits, complex hierarchical interactions, or mixed data types. Here, the dataset is small, numeric, and already well behaved. That often favors compact linear structure.

---

## Why Ridge And Elastic Net Were Included

Even when plain linear regression works well, regularized versions are still useful.

## Ridge Regression

Ridge applies an L2 penalty to the regression coefficients.

Optimization form:

`min_beta [ error term + lambda * L2 penalty ]`

where the L2 penalty is the sum of squared coefficients.

The reason for including Ridge is that the engineered features introduce correlation among predictors. For example, original features and their products or squares are naturally related. Ridge helps stabilize coefficients in that setting.

## Elastic Net

Elastic Net combines L1 and L2 regularization.

Optimization form:

`min_beta [ error term + lambda_1 * L1 penalty + lambda_2 * L2 penalty ]`

where:
* L1 penalty is the sum of absolute coefficients
* L2 penalty is the sum of squared coefficients

The reason for including Elastic Net is that it can both shrink correlated variables and reduce the influence of weaker engineered features.

Even when it does not become the best single model, it can add diversity in an ensemble.


---

## Why Polynomial Ridge Was Tested

The notebook also tests a second-degree polynomial Ridge pipeline.

This is included to answer a specific question:

* does broad second-order expansion help more than a compact hand-built feature set?

The result in this notebook suggests that full polynomial expansion is too broad for the current data size and feature space. Instead of improving fit, it appears to introduce excess variance and noise.

That is a useful negative result. It shows that more features do not automatically produce better generalization.

---

## Why Residual Analysis Is Included

The residual section is one of the most useful diagnostic parts of the notebook.

Residuals are defined as:

$$
r_i = y_i - \hat{y}_i
$$

The notebook plots:

* residual distributions
* residuals versus predictions

These plots help inspect whether the best models are leaving behind systematic structure.

Residual analysis matters because it can answer questions such as:

* Are high scores consistently underpredicted?
* Are low scores consistently overpredicted?
* Is there remaining curvature not captured by the model?
* Are a few ranges dominating the error?

This section is also the natural bridge to future improvements, because new features should ideally be motivated by visible residual structure rather than by blind feature expansion.

---

## Why The Notebook Uses A Weighted Ensemble

After ranking all models, the notebook blends the top three candidates with a nonnegative weighted average.

The ensemble takes the form:

$$
\hat{y}_{\text{ens}} = w_1 \hat{y}_1 + w_2 \hat{y}_2 + w_3 \hat{y}_3
$$

with:

$$
w_1 + w_2 + w_3 = 1
$$

The reason for this step is practical:

* even similar linear-family models can make slightly different errors
* averaging can reduce variance
* a blended prediction can be more stable than one model alone

The notebook performs a grid search over weights and keeps the combination with the best OOF MAE.

This is a simple but effective way to extract additional value from closely related strong models without introducing a more complex stacking framework.

---

## Why The Final Prediction Uses Validation-Based Selection

The final notebook step compares:

* best single-model OOF MAE
* best ensemble OOF MAE

The prediction source is selected based on the lower validation error.

This is an important choice because it keeps model selection tied to the evaluation metric rather than to visual preference or model complexity.

The final output is then written into:

* `submission.csv`

with the required columns:

* `id`
* `final_score`

---

## What The Notebook Suggests About Further Improvement

The notebook already reveals several useful directions for future work.

### 1. Feature Pruning

Some engineered variables may be weaker than others, especially among:

* ratio features
* aggregate summary features
* broad polynomial expansion terms

A more compact feature set may improve stability.

### 2. Repeated Cross-Validation

The current notebook uses one shuffled 10-fold setup. A repeated-seed KFold design can provide a more stable estimate of which compact linear variant is truly strongest.

### 3. More Focused Regularization Search

The linear family already performs best, so a more careful search over:

* Ridge `alpha`
* Elastic Net `alpha`
* Elastic Net `l1_ratio`

is a natural next step.

### 4. Residual-Guided Feature Design

The most reliable new features are usually the ones suggested by residual patterns. If residuals show curvature with respect to one original feature, adding one specific term is often better than broad feature expansion.

### 5. Compact Formula-Style Modeling

Since the notebook indicates strong linear behavior, another promising path is to test compact score-style formulas built from:

* original variables
* a few interactions
* a few squared terms

This can sometimes perform very well in synthetic tabular competitions.

---

## Closing Remarks

This notebook demonstrates a structured approach to student final score prediction through careful data inspection, broad exploratory analysis, hypothesis-driven feature engineering, regression benchmarking, and OOF-based ensembling.

The notebook is built to answer a practical modeling question: whether this competition is best approached through broad nonlinear complexity or through compact linear structure enriched by selected interactions.

The results support the second view. The strongest performance comes from the linear family, the residual analysis provides room for targeted refinement, and the ensemble step adds a stable final layer for submission.

Further improvements can be pursued through tighter feature selection, repeated-seed validation, regularization tuning, and compact formula-oriented feature design.

---

### Amey Thakur

[Kaggle](https://www.kaggle.com/ameythakur20) • [GitHub](https://github.com/Amey-Thakur)

---

<div align="center">

  [↑ Back to Top](#readme-top) &nbsp;·&nbsp; [← Back to Home](../README.md)

</div>
