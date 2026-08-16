<div align="center">

# AI Mathematical Olympiad: Notation-Aware Diagnostics & Inference Scaffold

**A modular agentic inference framework using RAG-backed symbolic computation and LaTeX-aware text diagnostics.**

<br>

[![Notebook](https://img.shields.io/badge/Notebook-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/ameythakur20/aimo-diagnostics-inference) [![Author](https://img.shields.io/badge/Author-Amey_Thakur-0969DA)](https://www.kaggle.com/ameythakur20)

<br>

<a href="https://www.kaggle.com/code/ameythakur20/aimo-diagnostics-inference"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open in Kaggle"></a>

<br>

**2 notebooks in this folder**

[`aimo-diagnostics-inference.ipynb`](./aimo-diagnostics-inference.ipynb) &nbsp;·&nbsp; [`aimo-setup.ipynb`](./aimo-setup.ipynb)

<br>

[Competitions](../../README.md#competitions) &nbsp;·&nbsp; [Achievements](../../Achievements/Badges/README.md) &nbsp;·&nbsp; [Courses](../../Kaggle%20Courses/README.md) &nbsp;·&nbsp; [Kaggle Profile](https://www.kaggle.com/ameythakur20)

</div>

---

## The problem

The AI Mathematical Olympiad Progress Prize 3 asks a model to solve competition
mathematics, where an answer is either right or it is wrong and there is no
partial credit. The reasoning is only half the difficulty. Problems arrive as
LaTeX, so the notation has to survive every processing stage intact: a
normalisation step that quietly mangles a delimiter changes the question being
answered.

## What is here

The work is split across two notebooks, both in this folder.

**Environment setup.** Submissions run with the internet disabled, so the
dependency tree has to be staged in advance. This notebook resolves pinned
versions of Unsloth, TRL and vLLM together with their binary trees, fetches the
tiktoken encodings needed to initialise a tokenizer locally, and packs the lot
into a single archive that can be installed offline.

**Diagnostics and inference.** The second notebook is the pipeline itself.

| Stage | What it does |
| :--- | :--- |
| Data acquisition | Registers the competition paths and stages model artefacts from local mounts |
| Data inspection | Audits the problem schema across splits for missing values, duplicates and formatting |
| Data cleaning | Normalises LaTeX and whitespace while holding symbols and delimiters stable |
| Exploratory analysis | Describes the corpus before any feature is derived from it |
| Feature engineering | Builds notation-aware features from the cleaned text |
| Inference | Runs the staged transformer stack against the evaluation harness |

> [!NOTE]
> The offline constraint is the reason setup is a notebook of its own rather
> than a cell at the top of the pipeline. Anything not present in the archive
> when the submission starts is simply unavailable.

**Stack** &nbsp;·&nbsp; `transformers` `scikit-learn` `pandas` `numpy`
`matplotlib` `seaborn` `kaggle_evaluation`

---

<div align="center">

**Amey Thakur** &nbsp;·&nbsp; [Kaggle](https://www.kaggle.com/ameythakur20) &nbsp;·&nbsp; [GitHub](https://github.com/Amey-Thakur) &nbsp;·&nbsp; [ORCID](https://orcid.org/0000-0001-5644-1575)

<br>

[↑ Back to top](#ai-mathematical-olympiad-notation-aware-diagnostics-inference-scaffold) &nbsp;·&nbsp; [← Repository home](../../README.md)

</div>
