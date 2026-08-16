<div align="center">

# LLM Classification Finetuning

**Ensembled Pipeline Inference for Human Preference Classification**

<br>

[![Notebook](https://img.shields.io/badge/Notebook-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/ameythakur20/llm-classification-inference) [![Author](https://img.shields.io/badge/Author-Amey_Thakur-0969DA)](https://www.kaggle.com/ameythakur20)

<br>

<a href="https://www.kaggle.com/code/ameythakur20/llm-classification-inference"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open in Kaggle"></a>

<br>

[Competitions](../../README.md#competitions) &nbsp;·&nbsp; [Achievements](../../Achievements/Badges/README.md) &nbsp;·&nbsp; [Courses](../../Kaggle%20Courses/README.md) &nbsp;·&nbsp; [Kaggle Profile](https://www.kaggle.com/ameythakur20)

</div>

---

## The problem

Given a prompt and two model responses, predict which one a human preferred.
The difficulty is positional bias: a model that reads response A first will
tend to favour it, and that preference has nothing to do with the responses
themselves.

## What is here

The countermeasure is to run the same data twice, once in each order, and
average the results. Each pass uses a different backbone, so the ensemble also
averages over architecture.

| Stage | What it does |
| :--- | :--- |
| Data structuring | Builds two views of the test set, one standard and one with Response A and Response B interchanged |
| Gemma-2 pass | Runs Gemma-2-9B with manual layer allocation across two GPUs and a variable-length collator |
| Llama-3 pass | Repeats the partitioning for Llama architecture dimensions, run against the swapped view |
| Ensemble | Averages the two passes, cancelling the order effect that either pass would carry alone |

> [!IMPORTANT]
> Inference runs with the internet disabled. The model weights and the
> `human_pref` helper module are attached as Kaggle datasets rather than
> installed, and the notebook expects them to be present in the environment.

**Stack** &nbsp;·&nbsp; `torch` `transformers` `xformers` `pandas` `numpy`

---

<div align="center">

**Amey Thakur** &nbsp;·&nbsp; [Kaggle](https://www.kaggle.com/ameythakur20) &nbsp;·&nbsp; [GitHub](https://github.com/Amey-Thakur) &nbsp;·&nbsp; [ORCID](https://orcid.org/0000-0001-5644-1575)

<br>

[↑ Back to top](#llm-classification-finetuning) &nbsp;·&nbsp; [← Repository home](../../README.md)

</div>
