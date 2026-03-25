"""
Kaggle Toolbox Demo: Bangalore House Prices
--------------------------------------------
A practical walkthrough of all 10 functions in the kaggle_toolbox
utility script, applied to a real-world housing dataset.

This notebook exists for two reasons:
  1. To prove that every function in the toolbox works end-to-end.
  2. To show you the exact order and context in which each tool
     is most useful during a typical competition workflow.

Dataset: https://www.kaggle.com/datasets/ameythakur20/bangalore-house-prices
Utility: https://www.kaggle.com/code/ameythakur20/kaggle-toolbox

Author: Amey Thakur (https://www.kaggle.com/ameythakur20)
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


# ---------------------------------------------------------------------------
# STEP 0: IMPORT THE TOOLBOX
# ---------------------------------------------------------------------------
# Kaggle mounts utility scripts in a separate directory that Python does
# not search by default. Without this path addition, "import kaggle_toolbox"
# will fail with ModuleNotFoundError even though the script is attached
# to your notebook. This is the single most common import issue on Kaggle.

UTILITY_PATH = "/kaggle/usr/lib/ameythakur20/kaggle-toolbox"
if os.path.exists(UTILITY_PATH):
    sys.path.append(UTILITY_PATH)

import kaggle_toolbox as tb


# ---------------------------------------------------------------------------
# STEP 1: HARDWARE CHECK  [system_info]
# ---------------------------------------------------------------------------
# Kaggle silently assigns different hardware across sessions. A notebook
# that ran fine on a T4 GPU yesterday can crash today if it gets a P100
# with less VRAM, or a CPU-only instance. Checking upfront prevents
# surprises halfway through a 9-hour training run.

tb.system_info()


# ---------------------------------------------------------------------------
# STEP 2: LOCK REPRODUCIBILITY  [seed_everything]
# ---------------------------------------------------------------------------
# Without explicit seed locking, two identical runs of the same notebook
# can produce different scores. This happens because numpy, torch, and
# Python's own hash function all draw from separate random pools. A
# single call here locks all of them at once.

tb.seed_everything(42)


# ---------------------------------------------------------------------------
# STEP 3: LOCATE THE DATASET  [find_input]
# ---------------------------------------------------------------------------
# Competition datasets live in /kaggle/input/<slug>/, but the exact slug
# changes with every competition and every dataset version. Rather than
# hardcoding paths or running !ls in a cell, find_input walks all known
# Kaggle directories and returns the full path to any file matching
# your keyword.

input_files = tb.find_input("bengaluru_house_prices")
DATA_PATH = input_files[0] if input_files else "bengaluru_house_prices.csv"


# ---------------------------------------------------------------------------
# STEP 4: TIMED DATA LOADING  [timer]
# ---------------------------------------------------------------------------
# Kaggle kernels have strict time limits (9 hours for GPU, 12 for CPU).
# Wrapping each major step in a timer block builds a natural performance
# log. When you hit the wall at hour 8, you can look back and see exactly
# which step ate most of your budget.

with tb.timer("Loading CSV"):
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        # Fallback: generate synthetic data for local testing.
        # This ensures the demo runs anywhere, not just on Kaggle.
        np.random.seed(42)
        df = pd.DataFrame({
            "area_type": np.random.choice(
                ["Super built-up Area", "Built-up Area", "Plot Area"], 200
            ),
            "availability": np.random.choice(["Ready To Move", "19-Dec"], 200),
            "location": np.random.choice(["Whitefield", "Sarjapur Road"], 200),
            "size": np.random.choice(["2 BHK", "3 BHK", "4 BHK"], 200),
            "society": [None] * 80 + ["Crest"] * 120,
            "total_sqft": np.random.randint(600, 4000, 200).astype(float),
            "bath": np.random.choice([1, 2, 3, 4], 200).astype(float),
            "balcony": np.random.choice([0, 1, 2, 3], 200).astype(float),
            "price": np.random.uniform(20, 600, 200),
        })

print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns.")


# ---------------------------------------------------------------------------
# STEP 5: MISSING VALUE AUDIT  [missing_report]
# ---------------------------------------------------------------------------
# df.isnull().sum() gives raw counts but no percentages, no dtypes, and
# no sorting. The missing_report function returns a proper DataFrame
# sorted by severity (highest missing percentage first), so the columns
# that need the most attention are immediately visible at the top.

print("\n--- Missing Value Audit ---")
missing = tb.missing_report(df)
print(missing)
print()


# ---------------------------------------------------------------------------
# STEP 6: MEMORY OPTIMIZATION  [reduce_mem_usage]
# ---------------------------------------------------------------------------
# Pandas defaults to int64 and float64 for all numeric columns. On a
# housing dataset with values like "bath = 2" and "price = 150", int64
# wastes 7 of its 8 bytes per cell. Multiplied across millions of rows,
# this is the difference between a kernel that runs and one that crashes
# with MemoryError. This function walks every column and picks the
# smallest dtype that still holds the actual data range.

df = tb.reduce_mem_usage(df)


# ---------------------------------------------------------------------------
# STEP 7: DEAD COLUMN DETECTION  [find_useless_columns]
# ---------------------------------------------------------------------------
# Wide datasets from feature stores often contain columns where every
# row has the same value (zero variance), or two columns that are
# identical under different names (duplicates). Both waste compute time
# during training and can mislead feature importance rankings. This
# check catches them before they pollute the pipeline.

useless = tb.find_useless_columns(df)
cols_to_drop = useless["constant"] + [pair[0] for pair in useless["duplicate"]]
if cols_to_drop:
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"Dropped {len(cols_to_drop)} useless column(s).")
else:
    print("No useless columns found. All features carry signal.")
print()


# ---------------------------------------------------------------------------
# STEP 8: CORRELATION FILTER  [find_correlated_features]
# ---------------------------------------------------------------------------
# Two features with 0.98 correlation carry nearly identical information.
# Tree-based models randomly split credit between them, which makes
# feature importance unstable and wastes splits. Dropping one of each
# correlated pair is standard practice in tabular competitions.

# Prepare numeric features for correlation analysis.
df = df.dropna(subset=["total_sqft", "bath", "price"])
df["total_sqft"] = pd.to_numeric(df["total_sqft"], errors="coerce")
df = df.dropna(subset=["total_sqft"])

numeric_features = ["total_sqft", "bath", "balcony"]
available = [c for c in numeric_features if c in df.columns]
X = df[available].copy()
y = df["price"].copy()

correlated = tb.find_correlated_features(X, threshold=0.90)
if correlated:
    X.drop(columns=correlated, inplace=True)
    print(f"Dropped {len(correlated)} correlated feature(s).")
print()


# ---------------------------------------------------------------------------
# STEP 9: CROSS-VALIDATION REPORT  [cv_score]
# ---------------------------------------------------------------------------
# sklearn's cross_val_score returns a raw numpy array with no context.
# During a competition you want to see each fold's score, the mean,
# the standard deviation, and the wall-clock time, all in one block.
# This wrapper prints a clean report and returns the scores array for
# downstream use (ensemble weighting, threshold tuning, etc.).

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

with tb.timer("5-Fold Cross-Validation"):
    scores = tb.cv_score(
        model, X, y,
        cv=5,
        scoring="neg_mean_absolute_error"
    )
    print(f"\nMean MAE: {-scores.mean():.2f} lakhs")
print()


# ---------------------------------------------------------------------------
# STEP 10: SUBMISSION SANITY CHECK  [check_submission]
# ---------------------------------------------------------------------------
# Kaggle limits daily submissions (usually 2-5). A malformed CSV with
# wrong columns, NaN predictions, or mismatched row counts wastes one
# of those slots and returns a cryptic "Scoring Error" with no detail.
# Running this check before uploading catches formatting bugs locally.

# Build a mock submission for demonstration purposes.
submission = pd.DataFrame({
    "Id": range(len(y)),
    "price": y.values
})
sample_submission = submission.copy()

print("--- Submission Validation ---")
tb.check_submission(submission, sample_submission)


# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
# This notebook exercised all 10 functions in the kaggle_toolbox:
#
#     1. system_info()              -- Know your hardware
#     2. seed_everything(42)        -- Eliminate phantom score changes
#     3. find_input("keyword")      -- Find any file Kaggle mounted
#     4. timer("description")       -- Track time budgets
#     5. missing_report(df)         -- Audit null values cleanly
#     6. reduce_mem_usage(df)       -- Prevent memory crashes
#     7. find_useless_columns(df)   -- Remove dead weight
#     8. find_correlated_features() -- Stabilize feature importance
#     9. cv_score(model, X, y)      -- Professional CV logging
#    10. check_submission(sub, ref) -- Protect your submission quota
#
# Import it into any notebook with:
#     import kaggle_toolbox as tb
