<div align="center">

# Write-up Standard

**How a competition folder explains itself: the README that sits beside the notebook, what it must answer, and how it is found.**

<br>

[![Audience](https://img.shields.io/badge/Audience-Humans_and_agents-0969DA)](#what-a-write-up-is-for)
[![Pairs with](https://img.shields.io/badge/Pairs_with-Notebook_Standard-20BEFF?logo=kaggle&logoColor=white)](NOTEBOOK-STANDARD.md)
[![Status](https://img.shields.io/badge/Status-Binding-2EA043)](#the-template)

<br>

[Repository home](../README.md) &nbsp;·&nbsp;
[Notebook Standard](NOTEBOOK-STANDARD.md) &nbsp;·&nbsp;
[Competitions](../README.md#competitions) &nbsp;·&nbsp;
[Kaggle Profile](https://www.kaggle.com/ameythakur20)

</div>

---


<a name="what-a-write-up-is-for"></a>
## What a write-up is for

The notebook shows **what** was done. The write-up exists to answer **why** it
was done that way and **how** the decision was reached. If the README only
narrates the notebook in prose, delete it: it has added a maintenance burden and
no information.

Three readers must be served by the same page:

| Reader | Arrives from | Wants |
| :--- | :--- | :--- |
| A practitioner | A search for the technique | The reasoning, fast, and whether it transfers |
| A recruiter or collaborator | The profile | Evidence of judgement, not of typing |
| A search engine | A query | Honest structure and specific language |

> [!IMPORTANT]
> The test for every paragraph: **could this sentence appear in any other
> competition's write-up?** If yes, it says nothing about this one. Cut it or
> make it specific.

<br>

<a name="the-template"></a>
## The template

Every competition README follows this frame exactly. The header and footer are
fixed so that nineteen folders read as one work rather than nineteen.

```markdown
<div align="center">

<!-- Medal block, only if one was earned. Above the title, image then caption. -->
<img src="../../Achievements/Medals/Bronze%20Medal.png" width="34" alt="Bronze medal">

**Bronze medal**

# <Competition name>

**<One line: the task, in plain words>**

<br>

[![Notebook](https://img.shields.io/badge/Notebook-Kaggle-20BEFF?logo=kaggle&logoColor=white)](<notebook url>)
[![Medal](https://img.shields.io/badge/Medal-Bronze-8E5B3D)](https://www.kaggle.com/ameythakur20/code)
[![Author](https://img.shields.io/badge/Author-Amey_Thakur-0969DA)](https://www.kaggle.com/ameythakur20)

<br>

**Notebook in this folder**

[`<file>.ipynb`](./<file>.ipynb)

<br>

<a href="<notebook url>"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open in Kaggle"></a>

<br>

[Competitions](../../README.md#competitions) &nbsp;·&nbsp; [Achievements](../../Achievements/Badges/README.md) &nbsp;·&nbsp; [Courses](../../Kaggle%20Courses/README.md) &nbsp;·&nbsp; [Kaggle Profile](https://www.kaggle.com/ameythakur20)

</div>

---

## The problem
## What is here
## <Method sections, named for what they do>
## Results
## Where it fails
## What would improve it

---

<div align="center">

**Amey Thakur** &nbsp;·&nbsp; [Kaggle](https://www.kaggle.com/ameythakur20) &nbsp;·&nbsp; [GitHub](https://github.com/Amey-Thakur) &nbsp;·&nbsp; [ORCID](https://orcid.org/0000-0001-5644-1575)

<br>

[↑ Back to top](#<title-anchor>) &nbsp;·&nbsp; [← Repository home](../../README.md)

</div>
```

> [!WARNING]
> Never place an image inside the `#` heading. GitHub strips the tag from the
> anchor it generates but keeps the space, producing a leading hyphen, and every
> back-to-top link pointing at that heading breaks silently.

<br>

<a name="sections"></a>
## What each section must answer

**The problem.** What is predicted, from what, and how it is scored. Then the
constraint that made it hard: no labels, CPU only, no internet, a metric that
punishes one component. Name it here, because everything below is a response to
it.

**What is here.** A table of stages, one row each, saying what the stage does.
Not what function was called.

**Method sections.** Named for the job, not numbered generically. "Temporal
localization" beats "Step 2". Each opens with the difficulty, then the choice
made, then what the choice costs. A method section with no cost stated is
incomplete: every choice trades something.

**Results.** The score, with the validation scheme named. Where local and public
disagree, say which was trusted. Report per-component figures when the metric
has components, because a composite hides which part failed.

**Where it fails.** Specific failure modes with their cause. "The model could be
improved" is not a failure mode. "The centroid drifts when several vehicles move
at once, because a weighted mean spreads across every active region" is.

**What would improve it.** Concrete, ordered by expected gain, with the reason
each would help. If one change dominates, say so.

<br>

<a name="seo"></a>
## Being found

Search rewards pages that answer a real question specifically. It is not a
separate activity from writing well; it is the same activity, done deliberately.

**The title is the query.** Use the competition's own name plus the method,
because that is what people search. `Petals to the Metal: Flower Classification
on TPU` finds readers. `My Solution` does not.

**The first paragraph carries the terms.** Someone searching *"Farneback optical
flow accident detection"* should meet those words in the opening lines, used
naturally, because that is genuinely what the page is about. Never repeat a
phrase to hit a count; that reads as spam to a person and is discounted by every
modern ranker anyway.

**Headings are the outline.** One `#` per page. `##` for real sections. Never
skip a level for visual effect. Search engines and screen readers both read the
heading tree as the document's structure, and a broken tree damages both.

**Alt text describes, it does not label.** `alt="chart"` helps nobody.
`alt="Distribution of accident times, peaking between five and ten seconds"`
serves a blind reader and describes the image to a crawler in one stroke.

**Name the tools in prose.** `XGBoost`, `CLIP ViT-B/32`, `Optuna`, `TPU v3-8`.
These are the terms people search for and they belong in sentences, not only in
a requirements list.

**Link out to what you used.** A link to the competition, the paper behind the
method, the library. Outbound links to authoritative sources are a signal of a
real document, and they help the reader.

**Every link resolves.** A dead link is worse than an absent one. Run the audit
before committing.

<br>

<a name="voice"></a>
## Voice

The same rules as the [Notebook Standard](NOTEBOOK-STANDARD.md#voice): formal,
plain, no banned vocabulary, no em dashes. Two additions specific to write-ups.

**Write the reasoning, not the narration.**

```
WRONG   We then applied feature engineering to create new features which
        improved model performance significantly.

RIGHT   Ratios beat raw counts here: household size varies by an order of
        magnitude, so spend per person separates the classes where total spend
        does not. That one change moved local CV from 0.71 to 0.78.
```

**Give the number that settles it.** "Improved performance" is unfalsifiable.
"0.71 to 0.78 on five-fold CV" is a claim a reader can check and reason about.

**Admit the ceiling.** Every approach has one. Naming it is what separates a
write-up from a sales page, and it is the part experienced readers look for
first.

<br>

<a name="checklist"></a>
## Before committing

- [ ] Header and footer match the template exactly
- [ ] Title carries the competition name and the method
- [ ] The constraint that made the problem hard is named in the first section
- [ ] Every method section states what the choice cost
- [ ] Results name the validation scheme
- [ ] Failure modes are specific and causal
- [ ] Every figure has descriptive alt text
- [ ] Every link resolves, notebook link included
- [ ] No sentence could be moved to another competition unchanged
- [ ] No banned word, no em dash, no image inside a heading

<br>

---

<div align="center">

**Amey Thakur** &nbsp;·&nbsp; [Kaggle](https://www.kaggle.com/ameythakur20) &nbsp;·&nbsp; [GitHub](https://github.com/Amey-Thakur) &nbsp;·&nbsp; [ORCID](https://orcid.org/0000-0001-5644-1575)

<br>

[↑ Back to top](#write-up-standard) &nbsp;·&nbsp; [← Repository home](../README.md)

</div>
