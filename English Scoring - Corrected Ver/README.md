<a name="readme-top"></a>
# English Scoring - Corrected Ver: Essay Scoring Transformation
### A predictive modeling approach using structured text features and ensemble regression to forecast student performance across seven scoring dimensions.

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/ameythakur20/english-scoring-regression)

---

# Hello fellow Kagglers!

This notebook presents a **compact and efficient regression pipeline** for the *English Scoring* competition. The objective is to predict **seven essay scoring targets** using a combination of structured text features, reduced TF-IDF representations, and target-wise ensemble models.

The solution integrates:

- structured linguistic features  
- word-level and character-level TF-IDF  
- dimensionality reduction using SVD  
- target-wise LightGBM and Ridge models  
- optimized blending of predictions  

Each component is designed to balance **performance, interpretability, and computational efficiency**.


---

## Understanding the Problem Setting

Each essay is evaluated across seven dimensions:

- Overall, Cohesion, Syntax, Vocabulary  
- Phraseology, Grammar, Conventions  

The task is a **multi-target regression problem**:

$$
\hat{y} \in \mathbb{R}^{7}
$$

### Evaluation Metric

Mean Columnwise Root Mean Squared Error:

$$
\text{MCRMSE} = \frac{1}{7} \sum_{i=1}^{7} \sqrt{\frac{1}{N} \sum (y_i - \hat{y}_i)^2}
$$

### Key challenge

- targets are correlated but not identical  
- textual signal is high-dimensional and sparse  
- need to balance semantic and structural signals  


---

## 1. Data Acquisition and Setup

Competition datasets are loaded from Kaggle input paths.

Configuration includes:

- feature limits for TF-IDF  
- SVD dimensions  
- cross-validation splits  

### Why this matters

Centralized configuration ensures reproducibility and consistent experimentation.


---

## 2. Data Inspection

The dataset contains:

- 5185 training samples  
- 1297 test samples  

Each sample includes:

- essay text  
- grade  
- prompt  
- seven scoring targets  

### Observations

- targets are continuous in range [1, 5]  
- multiple prompts introduce contextual variability  
- grade and prompt act as auxiliary signals  


---

## 3. Data Cleaning

Validation checks confirm:

- no missing values  
- no duplicate rows  
- all required columns present  

### Why this matters

Ensures model training is not affected by data inconsistencies.


---

## 4. Exploratory Data Analysis

### Target Distributions

Histograms show:

- centered distributions around ~3  
- moderate variance across targets  

### Correlation Analysis

From correlation matrix:

- strong inter-target correlation (≈ 0.65 to 0.81)  

### Implication

Shared structure exists, but independent modeling is still beneficial.


---

## 5. Feature Engineering

### Structured Features

Extracted features include:

- word_count, char_count  
- sentence_count, paragraph_count  
- lexical_diversity  
- punctuation_density  
- average lengths  

Example:

$$
\text{Lexical Diversity} = \frac{\text{Unique Words}}{\text{Total Words}}
$$

### Why this matters

These features capture:

- writing complexity  
- stylistic variation  
- structural quality  

---

### TF-IDF Representations

Two representations:

- word-level (1 to 2 grams, 35000 features)  
- character-level (3 to 5 grams, 30000 features)  

### Dimensionality Reduction

Applied using SVD:

- word SVD: 220 components  
- char SVD: 120 components  

Explained variance:

- word: ~0.21  
- char: ~0.33  

### Why this matters

- reduces sparsity  
- improves model efficiency  
- retains dominant semantic patterns  


---

## 6. Feature Assembly

Final feature matrix combines:

- numeric engineered features  
- encoded categorical features (grade, prompt)  
- reduced TF-IDF embeddings  

Final shapes:

- full model: 362 features  
- ridge model: 142 features  

### Why this matters

Different models benefit from different feature subsets.


---

## 7. Modeling Strategy

### Target-wise Modeling

Each target is modeled independently.

### Models used

- LightGBM (two configurations)  
- Ridge regression (scaled features)  

### Cross-validation

- StratifiedKFold using grade_prompt interaction  

### Why this matters

- target-wise approach handles subtle differences  
- stratification preserves distribution across folds  


---

## 8. Evaluation Metric Implementation

Custom MCRMSE function:

- computes RMSE per target  
- averages across all targets  

This ensures alignment with competition metric.


---

## 9. Ensemble and Blending

Three model outputs:

- LGBM_1  
- LGBM_2  
- Ridge  

### Blending

Grid search over weights:

$$
\hat{y}_{\text{ens}} = w_1 y_{lgbm1} + w_2 y_{lgbm2} + w_3 y_{ridge}
$$

Best weights:

- 0.3, 0.3, 0.4  

Best OOF score:

- 0.50656  

### Why this matters

Blending improves generalization by combining model strengths.


---

## 10. Performance Analysis

From evaluation:

- blended model outperforms individual models  
- consistent performance across folds (~0.50 to 0.52 MCRMSE)  

### Feature Importance

Top contributors include:

- SVD components  
- lexical features  
- punctuation-based features  

### Residual Analysis

Residual distribution shows:

- centered errors  
- no major systematic bias  


---

## Summary

The pipeline integrates:

1. structured linguistic feature extraction  
2. word and character TF-IDF representations  
3. SVD-based dimensionality reduction  
4. target-wise LightGBM and Ridge models  
5. stratified cross-validation  
6. optimized blending of predictions  

Each component contributes to capturing different aspects of essay quality.


---

## Closing Remarks

This notebook demonstrates that combining structured features with reduced text representations provides strong performance for essay scoring.

Further improvements can focus on:

- transformer-based embeddings for deeper semantic understanding  
- multi-task learning to leverage target correlations  
- advanced stacking instead of linear blending  
- feature selection for further dimensionality reduction  

These directions can further improve predictive performance and model robustness.


---

### Amey Thakur

[Kaggle](https://www.kaggle.com/ameythakur20) • [GitHub](https://github.com/Amey-Thakur)

---

<div align="center">

  [↑ Back to Top](#readme-top) &nbsp;·&nbsp; [← Back to Home](../README.md)

</div>
