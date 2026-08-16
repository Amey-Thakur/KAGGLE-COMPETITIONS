<div align="center">

# Notebook Standard

**How a competition notebook is written in this repository, and what an agent must produce to match it.**

<br>

[![Audience](https://img.shields.io/badge/Audience-Humans_and_agents-0969DA)](#how-to-use-this)
[![Toolbox](https://img.shields.io/badge/Toolbox-kaggle__toolbox-20BEFF?logo=kaggle&logoColor=white)](../Kaggle%20Toolbox/README.md)
[![Status](https://img.shields.io/badge/Status-Binding-2EA043)](#the-nine-rules)

<br>

[Repository home](../README.md) &nbsp;·&nbsp;
[Competitions](../README.md#competitions) &nbsp;·&nbsp;
[Toolbox](../Kaggle%20Toolbox/README.md) &nbsp;·&nbsp;
[Kaggle Profile](https://www.kaggle.com/ameythakur20)

</div>

---


<a name="how-to-use-this"></a>
## How to use this

Point any coding agent at this file before it writes or edits a notebook in this
repository. It is written to be read once and applied without further
instruction.

```
Read Standards/NOTEBOOK-STANDARD.md and follow it exactly.
Task: <the competition, the data, what to build>
```

The rules below are not preferences. A notebook that breaks one of them is
rejected and rewritten, so it is cheaper to satisfy them on the first pass.

> [!IMPORTANT]
> The single failure mode this document exists to prevent: a notebook that is
> long, fluent and empty. Length is not evidence of work. Every cell must earn
> its place by answering a question the previous cell raised.

<br>

<a name="the-nine-rules"></a>
## The nine rules

**1. Every number is justified where it appears.**
`n_splits=5`, `threshold=0.95`, `MAX_LEN=420` are decisions, not defaults. The
line that sets one says why that value and not another. If the answer is "it is
what the tutorial used", find the real answer or change the value.

**2. Comments say why, never what.**
The code already says what. A comment that restates it is noise that has to be
maintained. See [Comments](#comments) for the exact test.

**3. Markdown before code, always.**
Each code cell is preceded by prose stating the question it answers. A reader
scrolling only the markdown must be able to follow the whole argument without
reading a line of Python.

**4. Every figure carries a conclusion.**
A plot with no caption is a decoration. The caption states what the figure shows
and what was decided because of it. Axes are labelled with units.

**5. The notebook runs top to bottom on a fresh kernel.**
No cell depends on something defined below it or on a variable that only exists
because a cell was run twice. Restart and run all before every commit.

**6. Seeds are set once, at the top, through the toolbox.**
`tb.seed_everything(SEED)`. A result that cannot be reproduced is not a result.

**7. Nothing that failed is deleted silently.**
An approach that did not work is worth more than one that did, because it tells
the next reader where the floor is. Record it in one honest paragraph and move
on. Do not dress it up.

**8. No dead code.**
No commented-out blocks, no `print(df.head())` left from debugging, no cell that
exists because it might be useful later. Delete it. Git remembers.

**9. Local validation outranks the public leaderboard.**
A public score is computed on a fraction of the test set and rewards fitting its
noise. When the two disagree, say so in the notebook and explain which you
trusted and why.

<br>

<a name="structure"></a>
## Structure

Sections appear in this order. Skip one only when the competition makes it
meaningless, and say so rather than leaving a silent gap.

| # | Section | What belongs in it |
| :---: | :--- | :--- |
| 1 | **Problem** | What is predicted, from what, and how it is scored. The scoring metric shapes every later decision, so it is stated first. |
| 2 | **Setup** | Imports, `tb.seed_everything(SEED)`, paths, configuration constants in one block. |
| 3 | **Data** | Load, then describe shape, dtypes and missingness. `tb.missing_report()` and `tb.reduce_mem_usage()` belong here. |
| 4 | **Exploration** | Only the questions that change what is built next. Each figure answers one. |
| 5 | **Features** | What was constructed and the reasoning. Leakage is addressed explicitly, not assumed absent. |
| 6 | **Validation** | The split, and why that split suits this data. Time series, groups and imbalance each force a different choice. |
| 7 | **Model** | The approach, the alternatives considered, and why this one. |
| 8 | **Results** | Scores with the validation scheme named. Local against public if both exist. |
| 9 | **Error analysis** | Where it fails and what that reveals. The most valuable section and the one most often missing. |
| 10 | **Submission** | Build, then `tb.check_submission(sub, sample)` before writing the file. |
| 11 | **What would improve it** | Concrete next steps, ordered by expected gain. |

<br>

<a name="comments"></a>
## Comments

The test is simple: **delete the comment and read the line. If nothing was
lost, the comment was noise.**

```python
# WRONG - restates the code
df = df.drop_duplicates()          # drop duplicates
model.fit(X_train, y_train)        # fit the model
SEED = 42                          # set seed to 42

# RIGHT - supplies what the code cannot
df = df.drop_duplicates()          # 1,204 rows repeat verbatim; the organisers
                                   # confirmed this is an export artefact, not signal
model.fit(X_train, y_train)        # single fit: CV below showed fold variance
                                   # under 0.002, so bagging buys nothing here
SEED = 42                          # fixed across every run so the ablation table
                                   # below compares features, not seed luck
```

Comments that earn their place answer one of these:

- Why this value, and what happens at other values
- Why this approach rather than the obvious alternative
- What non-obvious property of the data forced this
- What breaks if someone changes this line

<br>

<a name="visualisation"></a>
## Visualisation

Each figure answers a stated question and ends with what it settled.

```python
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(train["accident_time"], bins=48, color="#20BEFF", edgecolor="white")
ax.set_xlabel("Accident time (seconds from clip start)")
ax.set_ylabel("Videos")
ax.set_title("Where the collision falls within a clip")
plt.tight_layout()
```

> Most collisions land in the first ten seconds, with a median of 6.9. A model
> that scans the whole clip uniformly spends most of its budget where the event
> almost never is, which is why the search window below is centred early rather
> than at the midpoint.

That paragraph is the point of the figure. Without it the plot is decoration.

**Rules.** Label both axes with units. Use one accent colour, not a palette, and
only vary colour where it encodes something. Never rely on colour alone to carry
a distinction. Sort bars by value, never alphabetically, unless the category
order is itself meaningful.

<br>

<a name="toolbox"></a>
## Using the toolbox

Attach [`kaggle_toolbox`](../Kaggle%20Toolbox/README.md) as a notebook input and
import it once. Do not paste these helpers into the notebook; a fix then has to
be made in every copy.

```python
import kaggle_toolbox as tb

tb.seed_everything(42)                       # every RNG in one call
tb.system_info()                             # records the machine the run happened on

train = tb.reduce_mem_usage(train)           # typically 60-80% less RAM
tb.missing_report(train)                     # nulls per column, before deciding on imputation
tb.find_useless_columns(train)               # constant and near-constant columns
tb.find_correlated_features(train, 0.95)     # pairs above the threshold

with tb.timer("Feature engineering"):        # times a block, so slow steps are visible
    ...

tb.cv_score(model, X, y, cv=5)               # cross-validated score
tb.check_submission(sub, sample_sub)         # shape, columns and range, before writing
tb.find_input("train.csv")                   # locates attached inputs without guessing paths
```

> [!TIP]
> `tb.check_submission()` runs before every submission file is written, without
> exception. A submission rejected for a column name is a wasted day, and it is
> the single most common way a good notebook scores zero.

<br>

<a name="voice"></a>
## Voice

Formal, precise, plain. Short sentences. The reader is a competent practitioner
who has not seen this competition; explain the reasoning, not the basics.

**Never use these words.** They are the fingerprints of generated prose and they
carry no information:

`full` · `essential` · `delve` · `leverage` · `careful` ·
`pivotal` · `realm` · `reliable` (except as the statistical term) ·
`seamless` · `show` · `state-of-the-art` · `testament` ·
`use` · `cutting-edge` · `game-changing` · `unlock` · `harness` ·
`it is important to note` · `in today's world` · `is a`

**Never use an em dash.** A comma, a colon or a full stop is always available.

**Do not open a section by announcing it.** "In this section, we will explore
the data" says nothing. Start with the finding.

```
WRONG   In this section, we will perform a full exploratory analysis
        to gain valuable insights into the underlying data distribution.

RIGHT   Two of the five classes account for 62% of the training set, and the
        smallest has 66 examples. That imbalance decides the validation split
        below.
```

<br>

<a name="example"></a>
## A worked cell

The same step, written both ways.

**Rejected.**

```python
# Exploratory Data Analysis
# In this section we will perform full EDA to gain insights.

# import libraries
import pandas as pd
import numpy as np

# read the data
train = pd.read_csv('/kaggle/input/comp/train.csv')

# print the head
print(train.head())

# check for missing values
print(train.isnull().sum())

# fill missing values
train = train.fillna(0)
```

Nothing here was decided. Every comment restates its line, the paths are
hard-coded, and the last line silently changes the data on an assumption that is
never stated.

**Accepted.**

> ### Which columns can be trusted?
>
> Before anything is built, the question is which of the 41 columns carry signal
> and which are artefacts of how the organisers exported the data.

```python
import kaggle_toolbox as tb

train = pd.read_csv(tb.find_input("train.csv"))
train = tb.reduce_mem_usage(train)           # 41 float64 columns, 2.1 GB to 480 MB

tb.missing_report(train)
```

> Three columns are more than 90% null and all three are post-outcome fields
> that will not exist at inference. They are dropped rather than imputed:
> filling them would manufacture a signal the test set cannot have.

```python
LEAKY = ["settled_at", "final_amount", "closed_reason"]
train = train.drop(columns=LEAKY)            # populated only after the label is known

# Zero is a real reading for these sensors, so a null cannot be filled with it.
# The median is used and a flag retained, because whether a sensor reported at
# all turned out to separate the classes better than the reading itself.
for col in SENSORS:
    train[f"{col}_missing"] = train[col].isna().astype("int8")
    train[col] = train[col].fillna(train[col].median())
```

The difference is not length. It is that a reader now knows what was decided,
what it cost, and what would break if they changed it.

<br>

<a name="checklist"></a>
## Before committing

- [ ] Restart and run all completes on a fresh kernel, top to bottom
- [ ] `tb.seed_everything()` is called once, before anything random
- [ ] Every constant carries its reason
- [ ] No comment restates its line
- [ ] Every figure has labelled axes and a caption stating the conclusion
- [ ] Error analysis is present and specific
- [ ] `tb.check_submission()` runs before the file is written
- [ ] No commented-out code, no debug prints, no empty cells
- [ ] No banned word, no em dash
- [ ] The folder `README.md` matches the notebook and links it

<br>

---

<div align="center">

**Amey Thakur** &nbsp;·&nbsp; [Kaggle](https://www.kaggle.com/ameythakur20) &nbsp;·&nbsp; [GitHub](https://github.com/Amey-Thakur) &nbsp;·&nbsp; [ORCID](https://orcid.org/0000-0001-5644-1575)

<br>

[↑ Back to top](#notebook-standard) &nbsp;·&nbsp; [← Repository home](../README.md)

</div>
