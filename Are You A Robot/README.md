<a name="readme-top"></a>
# Are You A Robot?
### Identifying AI-Generated Discourse through Stochastic Analysis

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/ameythakur20/are-you-a-robot)

---

# Hello fellow Kagglers!

This notebook presents a **multi-task pipeline for essay evaluation** developed for the *Are You A Robot?* competition. The solution addresses three independent subtasks:

- readability estimation  
- sentiment regression  
- human vs AI text detection  

Each subtask is solved using a method aligned with its structure. The notebook focuses on how each signal is computed, how features are constructed, and why each method is appropriate.

---

## Understanding the Problem Setting

Each essay must produce three outputs:

- **Subtask 1**: Readability score  
- **Subtask 2**: Sentiment score  
- **Subtask 3**: Human vs AI classification  

These subtasks differ in nature:

- readability depends on linguistic structure  
- sentiment depends on semantic content  
- detection depends on writing style  

A single modeling approach would introduce unnecessary complexity. The pipeline separates them to ensure correctness and interpretability.

---

## Subtask 1: Readability Estimation

### Metric Definition

Readability is computed using the Flesch Reading Ease formulation:

$$
\text{Score} = 206.835 - 1.015 \cdot \frac{W}{S} - 84.6 \cdot \frac{SYL}{W}
$$

Where:

- `W` = number of words  
- `S` = number of sentences  
- `SYL` = number of syllables  

The score is clipped to the range `[0, 100]`.

---

### Why this formulation

The metric captures two components:

- `W / S` measures sentence complexity  
- `SYL / W` measures word difficulty  

Higher values increase reading difficulty, reducing the final score. The coefficients scale both effects based on empirical linguistic studies.

---

### Syllable Counting Strategy

Accurate syllable estimation is critical.

A hybrid method is used:

- **CMU Pronouncing Dictionary** for known words  
- **vowel-group heuristic** for unknown words  

Steps:

1. normalize word using regex  
2. check dictionary lookup  
3. fallback to vowel-based estimation  

This ensures both accuracy and coverage.

---

### Implementation Characteristics

- deterministic computation  
- linear complexity over tokens  
- no model dependency  

This is appropriate since readability is a defined linguistic metric.

---

## Subtask 2: Sentiment Regression

### Objective

Predict a continuous sentiment score from essay text.

---

### Feature Representation

#### TF-IDF Features

TF-IDF assigns weight:

$$
\text{TF-IDF}(t, d) = \text{TF}(t, d) \cdot \log\left(\frac{N}{DF(t)}\right)
$$

Where:

- `TF(t, d)` = term frequency  
- `DF(t)` = document frequency  
- `N` = total number of documents  

This highlights informative terms while reducing the influence of common words.

Both word-level and character-level representations are used.

---

#### Statistical Features

- word count  
- character count  
- average word length  
- lexical diversity  

These features capture structural properties of text.

---

### Model Architecture

An ensemble regression model is used:

- Ridge Regression  
- Gradient Boosting Regressor  
- LightGBM Regressor  

Combined using a Voting Regressor.

---

### Why this design

- Ridge handles sparse TF-IDF efficiently  
- boosting models capture nonlinear relationships  
- LightGBM improves performance and scalability  

The ensemble improves robustness and generalization.

---

### Pipeline Flow

1. generate TF-IDF features  
2. compute statistical features  
3. combine feature sets  
4. train ensemble model  
5. predict sentiment scores  

---

## Subtask 3: Human vs AI Detection

### Objective

Identify whether text is human-written or AI-generated without labeled data.

---

### Approach

The problem is treated as **unsupervised stylometric analysis**.

---

### Features Used

#### Sentence Length Variability

- human writing shows higher variance  
- AI-generated text is more uniform  

---

#### Lexical Diversity

$$
\text{Lexical Diversity} = \frac{\text{Unique Words}}{\text{Total Words}}
$$

- higher for human text  
- lower for AI due to repetition  

---

#### Structural Patterns

- punctuation usage  
- sentence consistency  
- repetition behavior  

---

### Why this approach

No labeled data is available. The method relies on measurable differences in writing patterns rather than supervised learning.

---

## Submission Construction

Each essay generates three outputs.

For each datapoint:

1. compute readability score  
2. predict sentiment score  
3. compute detection signal  

Each result is stored as a separate row.

---

### Output Format

The submission contains:

- `Id`  
- `subtaskID`  
- `datapointID`  
- `answer`  

The format strictly follows competition requirements.

---

## Summary

- readability is computed using a deterministic linguistic formula  
- sentiment is predicted using TF-IDF features and ensemble regression  
- detection is based on stylometric signals  

Each subtask is handled using a method aligned with its structure, ensuring clarity and correctness.

---

## Closing Remarks

This notebook demonstrates that different problem types require different solution strategies. Separating deterministic computation, supervised learning, and unsupervised analysis results in a clean and interpretable pipeline.

Further improvements can focus on:

- improving syllable estimation for more stable readability scores  
- incorporating contextual embeddings for stronger sentiment modeling  
- refining stylometric features for more reliable AI detection  
- tuning model parameters and feature sets for better generalization  

These refinements can improve overall performance and robustness.

---

### Amey Thakur

[Kaggle](https://www.kaggle.com/ameythakur20) • [GitHub](https://github.com/Amey-Thakur)

---

<div align="center">

  [↑ Back to Top](#readme-top) &nbsp;·&nbsp; [← Back to Home](../README.md)

</div>
