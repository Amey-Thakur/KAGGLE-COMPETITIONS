<div align="center">

<img src="../../Achievements/Medals/Bronze%20Medal.png" width="34" alt="Bronze medal">

**Bronze medal**

# Kaggriculture

**An industrial multi-worker livestock agent and predictive market-order prioritization engine in a two-player farming simulation.**

<br>

[![Notebook](https://img.shields.io/badge/Notebook-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/ameythakur20/kaggriculture-premium-first-market-agent) [![Medal](https://img.shields.io/badge/Medal-Bronze-8E5B3D)](https://www.kaggle.com/ameythakur20/code) [![Author](https://img.shields.io/badge/Author-Amey_Thakur-0969DA)](https://www.kaggle.com/ameythakur20) [![ORCID](https://img.shields.io/badge/ORCID-0000--0001--5644--1575-A6CE39)](https://orcid.org/0000-0001-5644-1575) [![License](https://img.shields.io/badge/License-Apache_2.0-lightgrey)](https://www.apache.org/licenses/LICENSE-2.0)

<a href="https://www.kaggle.com/code/ameythakur20/kaggriculture-premium-first-market-agent"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open in Kaggle"></a>

<br>

**Notebooks in this folder**

[`kaggriculture-premium-first-market-agent.ipynb`](./kaggriculture-premium-first-market-agent.ipynb)

[`kaggriculture-deterministic-farm-planning-agent.ipynb`](./kaggriculture-deterministic-farm-planning-agent.ipynb)

<br>

[Competitions](../../README.md#competitions) &nbsp;·&nbsp; [Achievements](../../Achievements/Badges/README.md) &nbsp;·&nbsp; [Courses](../../Kaggle%20Courses/README.md) &nbsp;·&nbsp; [Kaggle Profile](https://www.kaggle.com/ameythakur20)

</div>

---

## 1. Problem Formulation & Competition Objective

Kaggriculture is a two-player turn-based economy simulation spanning 30 in-game days (720 discrete turns at 24 turns per day). Both agents start with $3,000 initial bank capital and a 10x10 farm grid where only the northwest 5x5 quadrant is active. The objective is terminal capital accumulation: maximizing bank balance at turn 720.

Success in Kaggriculture requires balancing three competing subsystems:
1. **Labor Throughput Scaling:** Deploying farm hands along the Fibonacci hiring cost curve ($143/day for 10 hands), scaling labor capacity 11x from 24 to 264 operations per day.
2. **Biological Asset Lifecycles:** Sustaining compounding cash flows from livestock (8 Cows and 4 Sheep) supported by dedicated wheat cycles rather than decaying one-time crops.
3. **Sequential Market Execution:** Preventing cheaper agricultural goods from absorbing high-multiplier town consumption capacity ahead of high-value livestock products.

---

## 2. Livestock Economics & Herd Scaling

Livestock assets produce compounding cash flows throughout the 30-day season when supplied with daily wheat feed:

$$\text{Milk Revenue} = 8 \text{ cows} \times 11 \text{ production cycles} \times \$160 = \$14,080 / \text{cycle cluster}$$

$$\text{Wool Revenue} = 4 \text{ sheep} \times 8 \text{ production cycles} \times \$200 = \$6,400 / \text{cycle cluster}$$

| Asset System | Lifespan | Daily Feed Requirement | Gross Revenue ($) | Net Realized Margin ($) |
| :--- | :---: | :---: | :---: | :---: |
| **8 Cows + 4 Sheep (Industrial)** | **Indefinite** | **Yes (Wheat)** | **$96,000** | **$88,400** |
| **Melon Matrix (Single Horizon)** | 10 Days | No (Water) | $28,000 | $24,200 |
| **Carrot Quick-Cycle** | 3 Days | No (Water) | $9,400 | $7,800 |
| **Wheat Baseline Loop** | 4 Days | No (Water) | $5,200 | $4,100 |

---

## 3. Market Priority Execution & Front-Running Discovery

Market sale transactions settle sequentially within the `market` command array. When lower-tier crops (Wheat and Carrots) are submitted prior to premium goods, they absorb town demand quotas at compressed unit prices, reducing multipliers for subsequent high-value goods.

To eliminate demand dilution, the agent enforces a priority reordering filter (`_reorder_market`):
- Premium items (`MELON`, `MILK`, `WOOL`, `STRAWBERRY`) execute first, capturing top town multipliers.
- Bulk staple crops execute second, clearing residual quantities.
- Dynamic front-running (`_front_run`) preempts town consumption steps, avoiding price degradation.

---

## 4. Agent Architecture & Runtime Subsystems

The production pipeline operates through deterministic state controllers:

1. **Closed-Loop Weed Repair (`_weed_repair_action`):** Tracks and patches stochastic weed spawns without disturbing the global action timeline.
2. **Predictive Inventory Front-Running (`_front_run`):** Submits pre-allocated sell orders exactly one step prior to town consumption ticks.
3. **Market Priority Arbitrator (`_reorder_market`):** Sorts market actions to guarantee premium asset execution first.
4. **Worker Hand Alignment (`_align_hands`):** Validates and pads hand orders to prevent execution mismatch.

---

## 5. Head-to-Head Tournament Benchmarks

Across local 720-step season simulations evaluated with official `kaggle_environments` (1.32.7) against established public baselines:

| Contender Agent | Champion (`Multi-Route V113`) | Record (W-L-T) | Win Rate | Mean Margin Alpha |
| :--- | :--- | :---: | :---: | :---: |
| **`Kaito-V41-Sparse`** | `Multi-Route V113` | **0W - 10L - 0T** | **100.0%** | **+$8,334.90** |
| **`Soil-Remembers-Rain (V26-H)`** | `Multi-Route V113` | **0W - 10L - 0T** | **100.0%** | **+$7,845.70** |
| **`Tetsutani-Adaptive`** | `Multi-Route V113` | **1W - 9L - 0T** | **90.0%** | **+$3,878.20** |
| **`V111-Baseline`** | `Multi-Route V113` | **1W - 5L - 0T** | **83.3%** | **+$6,012.70** |
| **`Starter Baseline`** | `Multi-Route V113` | **0W - 10L - 0T** | **100.0%** | **+$177,807.00** |

---

<div align="center">

**Amey Thakur** &nbsp;·&nbsp; [Kaggle](https://www.kaggle.com/ameythakur20) &nbsp;·&nbsp; [GitHub](https://github.com/Amey-Thakur) &nbsp;·&nbsp; [ORCID](https://orcid.org/0000-0001-5644-1575)

<br>

[↑ Back to top](#kaggriculture) &nbsp;·&nbsp; [← Repository home](../../README.md)

</div>