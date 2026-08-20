<div align="center">

<img src="../../Achievements/Medals/Bronze%20Medal.png" width="34" alt="Bronze medal">

**Bronze medal**

# Kaggriculture

**A multi-stage heuristic planning agent optimizing crop rotation economics, land acquisition, and dynamic market timing in a two-player farming simulation.**

<br>

[![Notebook](https://img.shields.io/badge/Notebook-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/ameythakur20/kaggriculture-deterministic-farm-planning-agent) [![Medal](https://img.shields.io/badge/Medal-Bronze-8E5B3D)](https://www.kaggle.com/ameythakur20/code) [![Author](https://img.shields.io/badge/Author-Amey_Thakur-0969DA)](https://www.kaggle.com/ameythakur20) [![ORCID](https://img.shields.io/badge/ORCID-0000--0001--5644--1575-A6CE39)](https://orcid.org/0000-0001-5644-1575) [![License](https://img.shields.io/badge/License-Apache_2.0-lightgrey)](https://www.apache.org/licenses/LICENSE-2.0)

<a href="https://www.kaggle.com/code/ameythakur20/kaggriculture-deterministic-farm-planning-agent"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open in Kaggle"></a>

<br>

**Notebook in this folder**

[`kaggriculture-deterministic-farm-planning-agent.ipynb`](./kaggriculture-deterministic-farm-planning-agent.ipynb)

<br>

[Competitions](../../README.md#competitions) &nbsp;·&nbsp; [Achievements](../../Achievements/Badges/README.md) &nbsp;·&nbsp; [Courses](../../Kaggle%20Courses/README.md) &nbsp;·&nbsp; [Kaggle Profile](https://www.kaggle.com/ameythakur20)

</div>

---

## 1. Executive Summary & Competition Objective

Kaggriculture is a two-player turn-based farming simulation spanning 30 in-game days (720 discrete turns). Each agent begins with an initial capital of $3,000 and a 10x10 farm grid where only the northwest 5x5 quadrant is active. The objective is strictly terminal capital accumulation: maximizing bank balance at the end of turn 720.

Success in Kaggriculture requires balancing three competing subsystems:
1. **Biological Growth Constraints:** Managing crop maturation cycles, daily watering schedules, and weed clearing penalties.
2. **Dynamic Price Elasticity:** Preventing market saturation where excessive dumping degrades unit sale prices toward the $1 floor.
3. **Temporal Horizon Boundaries:** Shifting from high-gestation capital-intensive crops (Melons) to rapid turnover staples (Wheat and Carrots) as the season reaches its cutoff.

---

## 2. Crop Economics & Strategic Selection

Each crop presents distinct capital requirements, maturation schedules, and yield ceilings. The table below compares the economic return per tile per day:

| Crop | Seed Cost ($) | Maturation (Days) | Gross Revenue ($) | Net Profit ($) | Daily Return ($/tile/day) | ROI Multiplier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Melon** | 80 | 10 | 1,500 | 1,420 | **$142.00** | 18.75x |
| **Wheat** | 10 | 4 | 100 | 90 | **$22.50** | 10.00x |
| **Carrot** | 20 | 3 | 105 | 85 | **$28.33** | 5.25x |
| **Tomato** | 50 | 11 | 240 | 190 | **$17.27** | 4.80x |
| **Strawberry** | 100 | 16 | 480 | 380 | **$23.75** | 4.80x |

### Policy Derivation
- **Days 1 to 5 (Working Capital Phase):** Fast Wheat rotations generate liquid funds with minimal downside risk.
- **Days 6 to 18 (Maximum Yield Phase):** Capital is deployed into Melons across all active tiles, capturing $142/tile/day return rates.
- **Days 19 to 30 (Harvest Liquidation Phase):** Planting of 10-day crops halts at Day 20. The agent transitions to Wheat and Carrots, completing full inventory liquidations prior to turn 720.

---

## 3. Dynamic Market Elasticity & Order Throttling

Market sale prices decline non-linearly as inventory accumulates above the baseline inventory ($I_0 = 100$). Selling large lots in a single turn drives realized revenue down exponentially.

To preserve profitability, the agent implements an order throttling mechanism:
- Inventory sales are executed in lots of 5 units.
- Sell orders trigger only when unit prices remain above $15, allowing town center consumption to clear market supply.
- Final inventory liquidation triggers on Day 28, converting all stored produce into bank balance regardless of price floors.

---

## 4. Agent Architecture & Decision Flow

The agent operates as a deterministic finite state machine with strict priority arbitration:

1. **Harvest Execution:** Prioritized to avoid lifespan decay and free tile capacity.
2. **Watering Schedule:** Evaluated every turn to prevent weed conversion and secure daily bonus yield multipliers.
3. **Land Acquisition:** Evaluates bank balance, unlocking the Northeast quadrant ($1,000) once liquid funds exceed $1,600.
4. **Seed Replenishment:** Purchases seed packages aligned with the current day horizon.
5. **Manhattan Pathing:** Directs farmer movement to the nearest tile requiring action.

---

## 5. Local Validation & Benchmark Trajectory

In local 30-day season simulations against a greedy baseline agent:

- **Strategic Planning Agent Final Capital:** **$11,840.00**
- **Greedy Baseline Final Capital:** **$6,580.00**
- **Performance Alpha:** **+80.0% capital advantage**

The primary advantage emerges during Days 10 to 20 when mature Melon harvests land and are liquidated at preserved price levels.

---

<div align="center">

**Amey Thakur** &nbsp;·&nbsp; [Kaggle](https://www.kaggle.com/ameythakur20) &nbsp;·&nbsp; [GitHub](https://github.com/Amey-Thakur) &nbsp;·&nbsp; [ORCID](https://orcid.org/0000-0001-5644-1575)

<br>

[Back to top](#kaggriculture) &nbsp;·&nbsp; [Repository home](../../README.md)

</div>