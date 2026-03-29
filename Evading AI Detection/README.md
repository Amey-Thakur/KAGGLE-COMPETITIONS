<a name="readme-top"></a>
# Evading AI-Generated Text Detection
### Activation Steering and Lexical Variance for Detection Evasion

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/ameythakur20/evading-ai-text-detection)

---

# Hello fellow Kagglers!

This notebook explores **methods for evading AI text detection systems** by modifying generated text while preserving semantic meaning. The objective is to analyze how stylistic transformations affect detection signals and to construct a pipeline that reduces detectability without degrading readability.

The focus is on controlled transformations that alter statistical properties of text while keeping the underlying content intact.


---

## Understanding the Problem Setting

AI text detectors identify generated content using statistical and stylometric signals such as:

- token probability patterns  
- sentence length distribution  
- lexical diversity  
- repetition and structural consistency  

AI-generated text tends to exhibit **low variance and predictable structure**, which makes it detectable.

The task in this notebook is to **shift these distributions** so that the text aligns more closely with human writing characteristics while preserving meaning.


---

## Core Strategy

The approach treats evasion as a **distribution shift problem**.

Instead of directly targeting a specific detector, the pipeline modifies text to:

- increase variability  
- reduce deterministic patterns  
- introduce controlled irregularity  

These changes weaken the statistical signals used by detection models.


---

## Text Transformation Pipeline

### 1. Lexical Variation

Words are replaced with semantically equivalent alternatives.

Objective:

- reduce repeated token usage  
- increase vocabulary diversity  

This directly alters token frequency distributions.

---

### 2. Sentence Structure Modification

Sentences are transformed by:

- varying sentence length  
- reordering clauses  
- introducing syntactic diversity  

Objective:

- increase variance in sentence structure  
- reduce uniformity typical of generated text  

---

### 3. Controlled Perturbation

Small changes are applied:

- synonym substitution  
- phrasing adjustments  

Objective:

- disrupt predictable generation patterns  
- simulate natural variation  

---

## Stylometric Effects

The transformations target measurable features used by detectors.

### Sentence Length Variance

Human writing exhibits higher variance, while generated text is more uniform. Increasing variance reduces detectability.

---

### Lexical Diversity

$$
\text{Lexical Diversity} = \frac{\text{Unique Words}}{\text{Total Words}}
$$

Higher diversity reduces repetition signals commonly associated with AI-generated text.

---

### Distribution Shift

The combined transformations move text away from:

- predictable token sequences  
- consistent structural patterns  

This reduces alignment with learned detector features.


---

## Trade-offs

There is a balance between:

- **semantic fidelity**  
- **degree of transformation**  

Excessive modification can distort meaning, while insufficient modification leaves detectable patterns unchanged.


---

## Evaluation Perspective

Evaluation is based on:

- reduction in detection confidence  
- preservation of semantic meaning  
- readability consistency  

The objective is not classification accuracy, but **effective distribution shift with minimal semantic loss**.


---

## Summary

- AI detection relies on stylometric and statistical regularities  
- generated text exhibits low variance and predictable structure  
- controlled transformations increase variability and disrupt these signals  
- effective evasion requires preserving meaning while altering distributional properties  


---

## Closing Remarks

This notebook demonstrates how modifying statistical properties of text can reduce detection signals without changing meaning.

Further improvements can focus on:

- learning transformation policies guided by detector feedback  
- using contextual rewriting models for more natural variations  
- quantifying semantic preservation during transformation  
- optimizing the balance between variability and readability  

These directions can improve evasion effectiveness while maintaining text quality.


---

### Amey Thakur

[Kaggle](https://www.kaggle.com/ameythakur20) • [GitHub](https://github.com/Amey-Thakur)

---

<div align="center">

  [↑ Back to Top](#readme-top) &nbsp;·&nbsp; [← Back to Home](../README.md)

</div>
