<div align="center">

# BirdCLEF+ 2026: Avian Vocalization Analysis & Perch v2 Fusion

**A high-performance bioacoustic pipeline using Perch v2 architecture and Bayesian model fusion for soundscape inference.**

<br>

[![Notebook](https://img.shields.io/badge/Notebook-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/ameythakur20/birdclef-2026-perch-v2-soundscape-inference) [![Author](https://img.shields.io/badge/Author-Amey_Thakur-0969DA)](https://www.kaggle.com/ameythakur20)

<br>

<a href="https://www.kaggle.com/code/ameythakur20/birdclef-2026-perch-v2-soundscape-inference"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open in Kaggle"></a>

<br>

[Competitions](../../README.md#competitions) &nbsp;·&nbsp; [Achievements](../../Achievements/Badges/README.md) &nbsp;·&nbsp; [Courses](../../Kaggle%20Courses/README.md) &nbsp;·&nbsp; [Kaggle Profile](https://www.kaggle.com/ameythakur20)

</div>

---

## The problem

BirdCLEF asks which species can be heard in a soundscape recording, in five
second windows, across a long tail of species for which very little audio
exists. Submissions run on CPU only and without internet, which rules out
training anything large at inference time and puts the weight on a strong
pretrained representation.

## What is here

**Environment setup.** Perch v2 ships as a SavedModel using StableHLO
operations introduced in TensorFlow 2.20, while the Kaggle image provides 2.19.
This notebook builds a portable wheelhouse of the exact TensorFlow and
`perch-hoplite` versions required, so the model can be served without an
upgrade at run time and without a network call.

**Perch v2 with Bayesian fusion.** The inference pipeline.

| Stage | What it does |
| :--- | :--- |
| Label preprocessing | Aggregates semicolon separated species codes per window, parses site, date and hour from the filename, and builds a multi-hot truth matrix |
| Species mapping | Maps each competition species onto its Perch class index, out of the 14,795 scientific names the model was pretrained on, leaving unmatched species explicitly unmapped |
| Embedding | Runs Perch v2 over the soundscapes on CPU, with seeds pinned and results cached |
| Fusion | Combines model outputs by Bayesian fusion, with the grid search frozen behind a mode switch so a submission run reuses fixed parameters rather than searching again |

> [!IMPORTANT]
> `CUDA_VISIBLE_DEVICES` is set to enforce CPU execution, which is a competition
> constraint rather than a preference. The caching layer exists because
> re-embedding the soundscapes on CPU is by far the most expensive step.

**Stack** &nbsp;·&nbsp; `tensorflow` `perch-hoplite` `scikit-learn` `scipy`
`soundfile` `pandas` `numpy` `kaggle_toolbox`

---

<div align="center">

**Amey Thakur** &nbsp;·&nbsp; [Kaggle](https://www.kaggle.com/ameythakur20) &nbsp;·&nbsp; [GitHub](https://github.com/Amey-Thakur) &nbsp;·&nbsp; [ORCID](https://orcid.org/0000-0001-5644-1575)

<br>

[↑ Back to top](#birdclef-2026-avian-vocalization-analysis-perch-v2-fusion) &nbsp;·&nbsp; [← Repository home](../../README.md)

</div>
