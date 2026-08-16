<div align="center">

# Agent Instructions

**Everything an agent needs to add a competition to this repository, with nothing left to judgement.**

<br>

[![Scope](https://img.shields.io/badge/Scope-Whole_repository-0969DA)](#the-job)
[![Notebook](https://img.shields.io/badge/Standard-Notebook-20BEFF?logo=kaggle&logoColor=white)](NOTEBOOK-STANDARD.md)
[![Write-up](https://img.shields.io/badge/Standard-Write--up-FAE041)](WRITEUP-STANDARD.md)

<br>

[Repository home](../README.md) &nbsp;·&nbsp;
[Notebook Standard](NOTEBOOK-STANDARD.md) &nbsp;·&nbsp;
[Write-up Standard](WRITEUP-STANDARD.md) &nbsp;·&nbsp;
[Toolbox](../Kaggle%20Toolbox/README.md)

</div>

---


Any coding agent reading this file, Claude Code, Cursor, Antigravity, Copilot or
otherwise, has everything required. Read the two standards it links before
writing anything.

<a name="the-job"></a>
## The prompt

Paste this, filling the four bracketed fields. Nothing else is needed.

```
You are adding a competition to Amey-Thakur/KAGGLE-COMPETITIONS.

Read these three files in full before writing anything:
  AGENTS.md
  Standards/NOTEBOOK-STANDARD.md
  Standards/WRITEUP-STANDARD.md

Competition : [name, exactly as Kaggle spells it]
Kaggle URL  : [competition link]
Notebook    : [your notebook URL, or "to be written"]
Goal        : [what to build, and any constraint: CPU only, no internet, offline]

Produce, in this order:
  1. Competitions/<Competition name>/<notebook-slug>.ipynb
  2. Competitions/<Competition name>/README.md
  3. The index row in README.md, regenerated, not hand-edited

Then run the verification in AGENTS.md and report the results. Do not claim
completion until every check passes. If something cannot be verified, say which
and why rather than asserting it.
```

<br>

<a name="layout"></a>
## Where things go

```
Competitions/<Competition name>/     # exactly as Kaggle spells it, spaces kept
    <notebook-slug>.ipynb            # lowercase, hyphens, matches the Kaggle slug
    README.md                        # the write-up
```

Never rename an existing folder. Paths are linked from the root index, from
Kaggle notebooks and from search results, and GitHub does not redirect a moved
path inside a repository.

<br>

<a name="colours"></a>
## Badge colours

Every colour is fixed. Kaggle's own values were sampled from the assets in this
repository rather than guessed, so do not substitute a visually similar hex.

| Concept | Hex | Where it comes from |
| :--- | :--- | :--- |
| Kaggle, Notebook | `20BEFF` | Kaggle brand blue |
| Notebooks Expert tier | `8148FD` | sampled from `Achievements/Tiers/Expert.png` |
| Medal, Medals | `8E5B3D` | sampled from `Achievements/Medals/Bronze Medal.png` |
| Courses | `FAE041` | sampled from the Kaggle Learn certificate seal |
| Badges | `BF3989` | house palette, research and focus |
| Status | `2EA043` | house palette, status |
| Author | `0969DA` | house palette, attribution |
| ORCID | `A6CE39` | ORCID brand |
| License | `lightgrey` | house palette, always grey |
| Counts | `3949AB` | house palette, quantities |

One colour means one thing. Before introducing a colour, check it is not already
in use for a different concept: `Courses` previously sat on `8250DF`, two
degrees from the Expert tier's `8148FD`, and the two badges read as identical on
the same row.

Silver is `C0C0C0` and gold is `FFD700` if either is ever earned.

**Exact badge markup.**

```markdown
[![Notebook](https://img.shields.io/badge/Notebook-Kaggle-20BEFF?logo=kaggle&logoColor=white)](<url>)
[![Medal](https://img.shields.io/badge/Medal-Bronze-8E5B3D)](https://www.kaggle.com/ameythakur20/code)
[![Author](https://img.shields.io/badge/Author-Amey_Thakur-0969DA)](https://www.kaggle.com/ameythakur20)
```

Underscores render as spaces. Encode `|` as `%7C`, `@` as `%40`, `:` as `%3A`.

<br>

<a name="never"></a>
## Fixed rules

**Never hardcode a count.** Not competitions, not notebooks, not medals held in
folders. The repository grows, and a number written into three files goes stale
the first time it does. The root index is generated from the folder listing.
Badge labels read `Badges-Earned` and `Courses-Kaggle_Learn`, not numbers.

**Kaggle standing is the one exception**, and it is supplied rather than
derived: Notebooks **Expert**, 7 bronze medals, highest rank **932** of 61,334.
Record only the highest rank ever reached, never the current one, which moves
whenever anyone publishes.

**Medals here are Notebooks medals**, awarded by community upvotes on published
notebooks. They are not competition placements. Titanic is a Getting Started
competition and awards no competition medals at all, so calling them placements
is factually wrong.

**Medal artwork is 32×32.** Display between 26 and 34 pixels. Larger blurs.

**No image inside a `#` heading.** GitHub strips the tag from the generated
anchor but keeps its space, producing a leading hyphen that silently breaks
every back-to-top link.

**Encode spaces in paths as `%20`**, and parentheses as `%28` `%29`. A folder
named `CS Week Codeathon AIML (Easy Level)` breaks a markdown link at the
closing bracket otherwise.

**No API token is needed.** Kaggle profiles, notebooks and competitions are
public and verify over plain HTTP. The `api/v1` endpoints return 401 without
auth and the profile page is client-rendered, but neither matters for checking a
link. Never write a credential to disk.

<br>

<a name="commits"></a>
## Commits

Subject is exactly `Kaggle Competitions`. No body, no description, no co-author
trailer. Commits are signed; signing is configured on the maintainer's side and
is not something an agent handles.

<br>

<a name="verify"></a>
## Verification

Run before reporting completion. A claim of "done" without these is not accepted.

| # | Check | Passing |
| :---: | :--- | :--- |
| 1 | Notebook restarts and runs top to bottom on a fresh kernel | no error |
| 2 | `tb.check_submission()` runs before any submission is written | present |
| 3 | Every relative link and image in the new README resolves | 0 broken |
| 4 | The Kaggle notebook URL returns HTTP 200 unauthenticated | 200 |
| 5 | Back-to-top anchor matches the anchor GitHub generates from the H1 | identical |
| 6 | No banned word, no em dash, in notebook or README | 0 |
| 7 | No commented-out code, no debug print, no empty cell | 0 |
| 8 | Root index regenerated, new competition present | present |
| 9 | Badge colours match the table above exactly | exact |
| 10 | No count hardcoded anywhere | 0 |

```bash
# 4: the notebook link must resolve for a signed-out visitor
curl -s -o /dev/null -w "%{http_code}\n" "https://www.kaggle.com/code/ameythakur20/<slug>"
```

> [!IMPORTANT]
> Report what actually happened. If a check fails, say so with the output. If a
> step was skipped, say it was skipped. A verification table filled in from
> expectation rather than from a run is worse than no table, because it removes
> the reason to look.

<br>

---

<div align="center">

**Amey Thakur** &nbsp;·&nbsp; [Kaggle](https://www.kaggle.com/ameythakur20) &nbsp;·&nbsp; [GitHub](https://github.com/Amey-Thakur) &nbsp;·&nbsp; [ORCID](https://orcid.org/0000-0001-5644-1575)

<br>

[↑ Back to top](#agent-instructions) &nbsp;·&nbsp; [← Repository home](../README.md)

</div>
