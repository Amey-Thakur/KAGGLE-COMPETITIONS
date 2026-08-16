<div align="center">

# Stanford RNA 3D Folding Part 2

**Structural Biology Pipeline Optimization**

<br>

[![Notebook](https://img.shields.io/badge/Notebook-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/ameythakur20/stanford-rna-3d-folding-part-2-tbm-protenix-v1) [![Author](https://img.shields.io/badge/Author-Amey_Thakur-0969DA)](https://www.kaggle.com/ameythakur20)

<br>

<a href="https://www.kaggle.com/code/ameythakur20/stanford-rna-3d-folding-part-2-tbm-protenix-v1"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open in Kaggle"></a>

<br>

**Notebook in this folder**

[`stanford-rna-3d-folding-part-2-tbm-protenix-gpu-t4.ipynb`](./stanford-rna-3d-folding-part-2-tbm-protenix-gpu-t4.ipynb)

<br>

[Competitions](../../README.md#competitions) &nbsp;·&nbsp; [Achievements](../../Achievements/Badges/README.md) &nbsp;·&nbsp; [Courses](../../Kaggle%20Courses/README.md) &nbsp;·&nbsp; [Kaggle Profile](https://www.kaggle.com/ameythakur20)

</div>

---

## The problem

Predict the three-dimensional coordinates of an RNA molecule from its sequence.
Structure prediction is expensive, and a T4 has limited memory, so the practical
question is where to spend that budget: on every target equally, or only on the
targets that need it.

## What is here

A two-phase hybrid that answers the second way.

| Phase | What it does |
| :--- | :--- |
| Template-based modelling | Aligns each target against the training database by pairwise sequence alignment. Where a template clears the identity threshold, the structure is taken from it |
| Protenix v1 | Everything without a usable template goes to the neural model, which is the expensive path |
| Sampling | Ten candidate models per target, to widen TM-score coverage rather than betting on a single prediction |

Two constants carry most of the tuning. `MAX_SEQ_LEN` is 420 tokens, chosen to
stay inside T4 memory while still covering the great majority of targets in one
pass, with longer sequences chunked. `N_SAMPLE` is 10.

> [!TIP]
> The exploratory section exists to justify those numbers rather than to
> decorate the notebook: the sequence length distribution is what shows where
> the chunking threshold has to sit.

**Stack** &nbsp;·&nbsp; `torch` `protenix` `biopython` `scipy` `pandas`
`numpy` `matplotlib` `seaborn`

---

<div align="center">

**Amey Thakur** &nbsp;·&nbsp; [Kaggle](https://www.kaggle.com/ameythakur20) &nbsp;·&nbsp; [GitHub](https://github.com/Amey-Thakur) &nbsp;·&nbsp; [ORCID](https://orcid.org/0000-0001-5644-1575)

<br>

[↑ Back to top](#stanford-rna-3d-folding-part-2) &nbsp;·&nbsp; [← Repository home](../../README.md)

</div>
