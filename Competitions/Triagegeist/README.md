<div align="center">

# Triagegeist: Hierarchical CDSS via Multi-Tier Acuity Forecasting (0.9995 CV)

**A reliable three-tier clinical decision support system using a blended meta-ensemble and uncertainty-aware safety logic for ESI triage.**

<br>

[![Notebook](https://img.shields.io/badge/Notebook-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/ameythakur20/triagegeist-cdss-multi-tier-acuity-forecasting) [![Author](https://img.shields.io/badge/Author-Amey_Thakur-0969DA)](https://www.kaggle.com/ameythakur20)

<br>

<a href="https://www.kaggle.com/code/ameythakur20/triagegeist-cdss-multi-tier-acuity-forecasting"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open in Kaggle"></a>

<br>

**Notebook in this folder**

[`Triagegeist_Clinical_Decision_Support_via_Multi_Tier_Acuity_Forecasting.ipynb`](./Triagegeist_Clinical_Decision_Support_via_Multi_Tier_Acuity_Forecasting.ipynb)

<br>

[Competitions](../../README.md#competitions) &nbsp;·&nbsp; [Achievements](../../Achievements/Badges/README.md) &nbsp;·&nbsp; [Courses](../../Kaggle%20Courses/README.md) &nbsp;·&nbsp; [Kaggle Profile](https://www.kaggle.com/ameythakur20)

</div>

---

This notebook presents a **Hierarchical Clinical Decision Support System (CDSS)** developed to categorize patient emergency levels using the Emergency Severity Index (ESI) framework.

The purpose of this explanation is to clarify how the three-tier hierarchy is structured, why specific physiological indicators are used, and how the **LightGBM-CatBoost** meta-ensemble ensures clinical stability.

Rather than focusing only on model training, the discussion follows the actual implementation in the notebook, covering data integration, hemodynamic feature science, specialist logic, and safety-aware ensembling.

---

### Project Description

Triagegeist is a clinical decision support system (CDSS) developed to categorize patient emergency levels using the Emergency Severity Index (ESI) framework. The system implements a three-tier hierarchical architecture that mirrors the multi-stage evaluation process in a hospital. This pipeline consists of an initial deterministic pattern recall, followed by specialized diagnostic models, and concluding with a weighted meta-ensemble of generalist models. By integrating high-dimensional text analysis with hemodynamic physiology, the system provides a reliable tool for assisting triage nurses in high-pressure clinical environments.

### Clinical Problem Statement

Emergency departments face a critical challenge in the initial minutes of patient contact: the risk of under-triage. When a patient with a life-threatening condition is incorrectly assigned a low-acuity score (e.g., ESI-4), essential interventions are delayed. Conversely, over-triage leads to resource exhaustion and increased wait times for the entire department. Current failure modes often stem from the high volume of subjective data in chief complaints and the physiological overlap between different acuity levels. Triagegeist targets this workflow gap by providing a mathematical anchor for triage decisions, ensuring that subtle but critical physiological indicators are prioritized during a brief intake assessment.

### Approach and Methodology

The approach combines three data streams: structured vital signs, longitudinal patient history, and unstructured chief complaint narratives. All tiers were implemented within a unified training pipeline and evaluated using stratified cross-validation.

1. **Data Integration and Imputation**: We joined disparate clinical tables on a unique patient identifier. Missing vital signs were handled using group-median imputation while preserving the original "missingness" signal through binary indicators, as absent data in an ER often correlates with high patient instability.

2. **Physiological Feature Science**: We derived specific hemodynamic indicators that provide higher predictive value than raw vitals. This includes Mean Arterial Pressure (MAP) to assess systemic perfusion, the Shock Index (Heart Rate/SBP) to detect early hemodynamic collapse, and Pulse Pressure to observe cardiac stroke volume trends.

3. **TF-IDF Text Analysis**: A high-dimensional vectorizer was implemented to translate qualitative nurse notes into quantitative signals. This allows the model to process symptoms like "Substernal Chest Pain" or "Sudden Onset Dyspnea" as distinct predictive features within the ensemble.

4. **The Triple-Tier Hierarchy**:
   * **Tier 1 (Recall)**: A deterministic lookup layer for unambiguous, high-acuity keywords.
   * **Tier 2 (Specialists)**: Targeted sub-models trained specifically on highly specific clinical subsets (e.g., ophthalmic variants like Glaucoma) to resolve niche clinical ambiguities.
   * **Tier 3 (Meta-Ensemble)**: A blended ensemble of LightGBM and CatBoost models (60/40 ratio) to finalize predictions for the remaining patient volume.

5. **Safety Guardrails**: We integrated a predictive entropy audit. If the system detects high uncertainty between two adjacent ESI classes, it defaults to the higher acuity level to prioritize patient safety and reduce under-triage.

### Results and Findings

The system achieved a reliable cross-validation Quadratic Weighted Kappa (QWK) score and high F1-macro metrics. Analysis of the Out-of-Fold (OOF) errors showed that the model is exceptionally precise at identifying ESI-1 (Resuscitation) and ESI-2 (Emergent) cases, which are the most time-sensitive categories. Failure modes occurred primarily in the distinction between ESI-3 and ESI-4, where clinical presentations are most similar. However, the safety-aware shifting logic successfully moved a majority of these uncertain cases into the higher-acuity group, effectively reducing the critical under-triage rate in the validation cohort.

### Limitations and Future Work

A primary limitation is that this model was trained on a specific institutional dataset. Clinical validation across multi-center cohorts is required to ensure the text vocabulary remains relevant across different regional charting styles. Additionally, while the model uses physiological indicators, it does not currently account for real-time telemetry trends. Future work will focus on integrating longitudinal monitoring data and expanding the specialist tier to include pediatric-specific triage logic.

### Reproducibility Notes

The following datasets from the Triagegeist competition are required: `train.csv`, `patient_history.csv`, and `chief_complaints.csv`. The finalized Kaggle Notebook runs end-to-end using a standard Python 3.10+ environment with `lightgbm`, `catboost`, and `scikit-learn`.
* **Random Seed**: All stochastic processes are locked to `42` for reproducibility.
* **Hardware**: Executable on standard GPU environments (P100/T4).
* **Memory**: The notebook uses optimized data types to remain within Kaggle resource limits.

---

## Closing Remarks

This notebook demonstrates a structured approach to clinical acuity prediction. By combining a multi-tier hierarchy with hemodynamic feature science and safety-aware ensembling, the pipeline achieves stable performance on both structured and text-based clinical data.

Further improvements can be explored through longitudinal vital sign monitoring and more granular specialist tiers for specific clinical domains.

---

<div align="center">

**Amey Thakur** &nbsp;·&nbsp; [Kaggle](https://www.kaggle.com/ameythakur20) &nbsp;·&nbsp; [GitHub](https://github.com/Amey-Thakur) &nbsp;·&nbsp; [ORCID](https://orcid.org/0000-0001-5644-1575)

<br>

[↑ Back to top](#triagegeist-hierarchical-cdss-via-multi-tier-acuity-forecasting-09995-cv) &nbsp;·&nbsp; [← Repository home](../../README.md)

</div>
