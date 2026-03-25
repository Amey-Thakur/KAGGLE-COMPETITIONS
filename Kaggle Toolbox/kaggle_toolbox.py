"""
kaggle_toolbox.py
-----------------
A daily-driver utility script for Kaggle competitors.

Every function here exists because it solves a problem that wastes
real time in real competitions. Nothing is decorative.


QUICK START
-----------
Step 1: Add this script as input to your notebook.
        Click "+ Add Input" > "Notebooks" > search "kaggle-toolbox"

Step 2: Import it at the top of your notebook:

        import kaggle_toolbox as tb

Step 3: Use any function you need. Here are the most common ones:

        tb.seed_everything(42)                              # Lock all random seeds
        train = tb.reduce_mem_usage(train)                  # Cut RAM usage by 60-80%
        tb.missing_report(train)                            # See which columns have nulls
        tb.check_submission(sub, sample_sub)                # Validate before uploading
        tb.system_info()                                    # Check your GPU, RAM, disk

        with tb.timer("Model training"):                    # Time any code block
            model.fit(X, y)

        tb.find_correlated_features(train)                  # Find redundant features
        tb.find_useless_columns(train)                      # Find zero-variance columns
        tb.cv_score(model, X, y, cv=5, scoring="accuracy")  # Clean CV report
        tb.find_input("train")                              # Find where Kaggle put your files


FULL FUNCTION LIST
------------------
 1. seed_everything(seed)          -- Reproducibility
 2. reduce_mem_usage(df)           -- Memory optimization
 3. missing_report(df)             -- Null value analysis
 4. find_useless_columns(df)       -- Constant and duplicate column detection
 5. check_submission(sub, sample)  -- Submission file validation
 6. timer(description)             -- Code block timing
 7. system_info()                  -- Hardware diagnostics
 8. cv_score(model, X, y)          -- Cross-validation reporting
 9. find_correlated_features(df)   -- Correlation-based feature filtering
10. find_input(pattern)            -- Kaggle path finder

Author: Amey Thakur (https://www.kaggle.com/ameythakur20)
"""

import os
import sys
import time
import random
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. REPRODUCIBILITY
# ---------------------------------------------------------------------------
# Kaggle scores fluctuate between identical runs when seeds are not locked
# across every source of randomness. A single missed seed (especially
# PYTHONHASHSEED or cudnn) causes phantom score differences that waste
# hours of debugging time.

def seed_everything(seed=42):
    """
    Lock every known source of randomness to guarantee identical results.

    Example:
        tb.seed_everything(42)
        # Now all random operations (numpy, torch, etc.) will produce
        # the same numbers every time you restart and run the kernel.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic mode trades a small amount of speed for exact
        # reproducibility. The tradeoff is worth it during development
        # because it eliminates a whole class of debugging dead ends.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    print(f"[toolbox] All seeds locked to {seed}")


# ---------------------------------------------------------------------------
# 2. MEMORY REDUCTION
# ---------------------------------------------------------------------------
# Pandas reads every integer as int64 and every float as float64 by default.
# On Kaggle's 13 GB RAM limit, a 2 GB dataset becomes an 8 GB DataFrame
# that crashes your kernel. Downcasting to the smallest type that still
# holds the actual data range cuts memory by 60-80% without losing any
# information. This is the single most copied function on Kaggle for
# good reason.

def reduce_mem_usage(df, verbose=True):
    """
    Downcast every numeric column to the smallest dtype that fits its values.

    Integer columns are checked against the actual min/max of the data
    and mapped to int8/int16/int32 accordingly. Float columns follow the
    same logic for float32. Object columns with low cardinality are
    converted to categorical. Datetime-like strings are auto-detected.

    Handles columns with missing values gracefully. If a column has NaN
    and cannot be safely cast to integer, it falls back to float32 which
    natively supports NaN without crashing.

    Example:
        train = pd.read_csv("train.csv")
        train = tb.reduce_mem_usage(train)
        # Output: [toolbox] Memory: 847.3 MB -> 214.1 MB (74.7% reduction)
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and col_type.name != "category":
            c_min = df[col].min()
            c_max = df[col].max()
            has_nulls = df[col].isnull().any()

            if str(col_type).startswith("int") or str(col_type).startswith("uint"):
                # Walk through integer types from smallest to largest.
                # We check boundaries explicitly because a silent overflow
                # (e.g., value 200 stored as int8 which maxes at 127) would
                # corrupt your data without any error message.
                #
                # If the column has NaN values (which can happen when pandas
                # reads mixed int/null columns as float), we skip integer
                # casting entirely and fall back to float32. Raw numpy int
                # types cannot hold NaN and will throw ValueError.
                if has_nulls:
                    df[col] = df[col].astype(np.float32)
                elif c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)

            elif str(col_type).startswith("float"):
                # Float16 has only ~3 decimal digits of precision, so we
                # skip it to avoid introducing rounding artifacts into
                # features or targets. Float32 with ~7 digits is the
                # practical minimum for ML work.
                if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)

        elif col_type == object:
            n_unique = df[col].nunique()
            n_total = len(df[col])

            # Categorical conversion is only beneficial when the number of
            # unique strings is small relative to total rows. A column with
            # 1 million unique strings out of 1 million rows gains nothing
            # from categorical encoding and actually wastes memory building
            # the category index.
            if n_unique / n_total < 0.5:
                df[col] = df[col].astype("category")
            else:
                # Attempt datetime parsing for columns that look like dates.
                # We only try if the column has string values that pandas can
                # recognize as timestamps. The errors="coerce" ensures we do
                # not crash on mixed-format columns.
                try:
                    sample = df[col].dropna().head(20)
                    parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
                    if parsed.notna().sum() > len(sample) * 0.8:
                        df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                except Exception:
                    pass

    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2

    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem if start_mem > 0 else 0
        print(f"[toolbox] Memory: {start_mem:.1f} MB -> {end_mem:.1f} MB ({reduction:.1f}% reduction)")

    return df


# ---------------------------------------------------------------------------
# 3. MISSING DATA REPORT
# ---------------------------------------------------------------------------
# df.info() prints to stdout in a format you cannot sort or filter.
# df.isnull().sum() gives counts but not percentages or dtypes.
# This function returns a clean DataFrame you can sort, filter, and
# display in a notebook cell, which makes it far more practical for
# deciding how to handle nulls during feature engineering.

def missing_report(df):
    """
    Returns a DataFrame showing missing count, percentage, and dtype
    for every column that has at least one null value.
    Sorted by missing percentage (highest first) so the worst offenders
    are immediately visible.

    Example:
        report = tb.missing_report(train)
        # Returns a DataFrame like:
        #               missing_count  missing_pct   dtype
        # cabin                  687        77.10  object
        # age                    177        19.87  float64
        # embarked                 2         0.22  object
    """
    total = df.isnull().sum()
    pct = 100 * total / len(df)
    dtypes = df.dtypes

    report = pd.DataFrame({
        "missing_count": total,
        "missing_pct": pct.round(2),
        "dtype": dtypes
    })

    # Only show columns that actually have missing values.
    # Showing all columns adds noise and makes real problems harder to spot.
    report = report[report["missing_count"] > 0]
    report = report.sort_values("missing_pct", ascending=False)

    if report.empty:
        print("[toolbox] No missing values found.")
    else:
        print(f"[toolbox] {len(report)} column(s) with missing values:")

    return report


# ---------------------------------------------------------------------------
# 4. CONSTANT AND DUPLICATE COLUMN DETECTION
# ---------------------------------------------------------------------------
# Columns with a single unique value carry zero predictive signal and
# waste memory, compute time, and feature importance budget. Duplicate
# columns (identical values under different names) do the same.
# Both are surprisingly common in competition datasets, especially those
# generated from wide joins or feature stores.

def find_useless_columns(df):
    """
    Identifies two categories of columns that should be dropped:
    1. Constant columns (only one unique value including NaN).
    2. Duplicate columns (same values as another column).

    Returns a dict with keys "constant" and "duplicate", each containing
    a list of column names. You can then do:

        df.drop(result["constant"], axis=1, inplace=True)

    Example:
        result = tb.find_useless_columns(train)
        # Output: [toolbox] Found 3 constant and 1 duplicate column(s)
        # result["constant"]  -> ["col_A", "col_B", "col_C"]
        # result["duplicate"] -> [("col_X", "col_Y")]  (col_X is a copy of col_Y)
    """
    result = {"constant": [], "duplicate": []}

    # Constant detection: nunique(dropna=False) counts NaN as a value,
    # which catches the edge case of a column that is entirely NaN
    # (nunique would be 0 with dropna=True, which is still useless).
    for col in df.columns:
        if df[col].nunique(dropna=False) <= 1:
            result["constant"].append(col)

    # Duplicate detection: compare each pair of columns by hashing their
    # actual content. We convert to string representation first because
    # raw .tobytes() on object columns hashes memory pointers instead of
    # the actual text, which would miss duplicate string columns entirely.
    remaining = [c for c in df.columns if c not in result["constant"]]
    seen = {}
    for col in remaining:
        col_key = hash(df[col].astype(str).str.cat())
        if col_key in seen:
            result["duplicate"].append((col, seen[col_key]))
        else:
            seen[col_key] = col

    n_const = len(result["constant"])
    n_dup = len(result["duplicate"])
    print(f"[toolbox] Found {n_const} constant and {n_dup} duplicate column(s)")

    return result


# ---------------------------------------------------------------------------
# 5. SUBMISSION VALIDATION
# ---------------------------------------------------------------------------
# Kaggle limits you to a fixed number of submissions per day (usually 5).
# A malformed CSV (wrong columns, NaN in predictions, mismatched row
# count) wastes one of those slots and returns a cryptic "Scoring Error."
# Validating locally before uploading saves both time and submission quota.

def check_submission(sub_df, sample_sub_df, check_range=True):
    """
    Validates a submission DataFrame against the sample submission.

    Pass your generated submission and the sample_submission.csv that
    Kaggle provides with every competition. This function checks:
    - Row count match
    - Column name and order match
    - Missing values in any column
    - Duplicate IDs
    - Value range sanity

    Returns True if all checks pass, False otherwise.

    Example:
        sample_sub = pd.read_csv("sample_submission.csv")
        my_sub = pd.read_csv("submission.csv")
        tb.check_submission(my_sub, sample_sub)
        # Output:
        # [toolbox] PASS: Row count (10000)
        # [toolbox] PASS: Columns match (2 cols)
        # [toolbox] PASS: No NaN values
        # [toolbox] PASS: No duplicate IDs
        # [toolbox] All checks passed. Safe to submit.
    """
    passed = True

    # Row count: must match exactly. Even one extra or missing row causes
    # a scoring failure on most competitions.
    if len(sub_df) != len(sample_sub_df):
        print(f"[toolbox] FAIL: Row count mismatch. Got {len(sub_df)}, expected {len(sample_sub_df)}")
        passed = False
    else:
        print(f"[toolbox] PASS: Row count ({len(sub_df)})")

    # Column check: names AND order must match because some competitions
    # use positional column indexing during scoring.
    if list(sub_df.columns) != list(sample_sub_df.columns):
        print(f"[toolbox] FAIL: Column mismatch.")
        print(f"  Got:      {list(sub_df.columns)}")
        print(f"  Expected: {list(sample_sub_df.columns)}")
        passed = False
    else:
        print(f"[toolbox] PASS: Columns match ({len(sub_df.columns)} cols)")

    # NaN check: almost no competition accepts NaN predictions.
    # The one exception would be explicit "no prediction" markers,
    # but those are rare enough that a warning is always appropriate.
    total_nan = sub_df.isnull().sum().sum()
    if total_nan > 0:
        nan_cols = sub_df.columns[sub_df.isnull().any()].tolist()
        print(f"[toolbox] FAIL: {total_nan} NaN values in columns: {nan_cols}")
        passed = False
    else:
        print(f"[toolbox] PASS: No NaN values")

    # Duplicate ID check: most competitions require one prediction per
    # sample. Duplicate IDs mean one sample gets predicted twice and
    # another gets skipped entirely, guaranteeing a low score.
    id_col = None
    for candidate in ["id", "ID", "Id", "index"]:
        if candidate in sub_df.columns:
            id_col = candidate
            break

    if id_col is not None:
        n_dup = sub_df[id_col].duplicated().sum()
        if n_dup > 0:
            print(f"[toolbox] FAIL: {n_dup} duplicate values in '{id_col}'")
            passed = False
        else:
            print(f"[toolbox] PASS: No duplicate IDs")

    # Range check: if predictions fall wildly outside the sample range,
    # it usually indicates a scaling bug or a log/exp that was forgotten.
    if check_range:
        numeric_cols = sub_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in sample_sub_df.columns and sample_sub_df[col].dtype in [np.float64, np.float32, np.int64]:
                s_min, s_max = sample_sub_df[col].min(), sample_sub_df[col].max()
                p_min, p_max = sub_df[col].min(), sub_df[col].max()

                # A 10x range violation is almost always a bug.
                s_range = max(abs(s_max - s_min), 1e-6)
                if p_min < s_min - 10 * s_range or p_max > s_max + 10 * s_range:
                    print(f"[toolbox] WARN: '{col}' range [{p_min:.4f}, {p_max:.4f}] "
                          f"far outside sample [{s_min:.4f}, {s_max:.4f}]")

    if passed:
        print("[toolbox] All checks passed. Safe to submit.")

    return passed


# ---------------------------------------------------------------------------
# 6. CODE TIMING
# ---------------------------------------------------------------------------
# Kaggle kernels have strict time limits (9 hours for GPU, 12 for CPU).
# Knowing exactly how long each section takes lets you decide where to
# optimize and whether you can afford another fold or a larger model.
# A context manager is the cleanest interface because it requires zero
# boilerplate and keeps the timing logic out of your main code.

@contextmanager
def timer(description="Block"):
    """
    Context manager that prints elapsed time for any code block.

    Wrap any section of code to see exactly how long it takes.
    Automatically formats seconds, minutes, or hours depending on duration.

    Example:
        with tb.timer("Loading data"):
            train = pd.read_csv("train.csv")
        # Output: [toolbox] Loading data: 3.2s

        with tb.timer("Training XGBoost"):
            model.fit(X_train, y_train)
        # Output: [toolbox] Training XGBoost: 4.7m
    """
    t0 = time.time()
    yield
    elapsed = time.time() - t0

    if elapsed < 60:
        print(f"[toolbox] {description}: {elapsed:.1f}s")
    elif elapsed < 3600:
        print(f"[toolbox] {description}: {elapsed / 60:.1f}m")
    else:
        print(f"[toolbox] {description}: {elapsed / 3600:.1f}h")


# ---------------------------------------------------------------------------
# 7. SYSTEM DIAGNOSTICS
# ---------------------------------------------------------------------------
# When your kernel crashes or behaves differently from someone else's
# fork, the first question is always "what hardware am I actually on?"
# Kaggle silently assigns different GPU models (T4, P100, sometimes
# even different CPU core counts) and the available RAM fluctuates.
# This function gives you immediate answers.

def system_info():
    """
    Print a quick summary of the current Kaggle environment.
    Shows Python version, CPU cores, RAM, GPU model, and free disk space.

    Example:
        tb.system_info()
        # Output:
        # ==================================================
        #  KAGGLE ENVIRONMENT REPORT
        # ==================================================
        # Python:   3.12.4
        # Platform: Linux x86_64
        # CPU:      4 cores
        # RAM:      12.7 GB
        # GPU:      Tesla T4, 15360 MiB
        # Disk:     14.2 GB free / 20.0 GB total
        # ==================================================
    """
    import platform

    print("=" * 50)
    print(" KAGGLE ENVIRONMENT REPORT")
    print("=" * 50)
    print(f"Python:   {platform.python_version()}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"CPU:      {os.cpu_count()} cores")

    # RAM: read from /proc/meminfo on Linux (Kaggle runs Ubuntu).
    # This is more reliable than psutil because psutil is not always
    # installed on Kaggle kernels.
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemTotal" in line:
                    mem_kb = int(line.split()[1])
                    print(f"RAM:      {mem_kb / 1024 / 1024:.1f} GB")
                    break
    except FileNotFoundError:
        print("RAM:      (not available on this OS)")

    # GPU: nvidia-smi is the most reliable way to identify the GPU
    # because torch.cuda.get_device_name() only works after CUDA
    # initialization, which we want to avoid triggering prematurely.
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print(f"GPU:      {result.stdout.strip()}")
        else:
            print("GPU:      None detected")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("GPU:      None detected")

    # Disk: Kaggle gives ~20 GB of working space. If you are saving
    # large model checkpoints or intermediate DataFrames, you can
    # silently run out and get cryptic I/O errors.
    try:
        import shutil
        total, used, free = shutil.disk_usage("/kaggle/working")
        print(f"Disk:     {free / 1024 ** 3:.1f} GB free / {total / 1024 ** 3:.1f} GB total")
    except FileNotFoundError:
        pass

    print("=" * 50)


# ---------------------------------------------------------------------------
# 8. QUICK CROSS-VALIDATION SUMMARY
# ---------------------------------------------------------------------------
# sklearn's cross_val_score returns a raw array. During a competition
# you want to see each fold's score, the mean, standard deviation,
# and total time, all printed cleanly in one block. This wrapper
# does exactly that.

def cv_score(model, X, y, cv=5, scoring=None, groups=None):
    """
    Runs cross-validation and prints a clean fold-by-fold report.

    Returns the array of fold scores so you can use it downstream
    (e.g., for ensemble weighting or threshold tuning).

    Parameters:
        model    -- Any sklearn-compatible estimator (XGBoost, LightGBM, etc.)
        X        -- Feature matrix (DataFrame or numpy array)
        y        -- Target vector
        cv       -- Number of folds (default: 5)
        scoring  -- Metric name, e.g. "accuracy", "roc_auc", "rmse"
                     Full list: https://scikit-learn.org/stable/modules/model_evaluation.html
        groups   -- Optional group labels for GroupKFold

    Example:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100)
        scores = tb.cv_score(model, X_train, y_train, cv=5, scoring="accuracy")
        # Output:
        # [toolbox] Cross-Validation Results (5 folds):
        #   Fold 1: 0.84200
        #   Fold 2: 0.85600
        #   Fold 3: 0.83100
        #   Fold 4: 0.86000
        #   Fold 5: 0.84500
        #   -------------------------
        #   Mean:   0.84680
        #   Std:    0.00964
        #   Time:   12.3s
    """
    from sklearn.model_selection import cross_val_score as _cvs

    t0 = time.time()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = _cvs(model, X, y, cv=cv, scoring=scoring, groups=groups)

    elapsed = time.time() - t0

    print(f"[toolbox] Cross-Validation Results ({cv} folds):")
    for i, s in enumerate(scores):
        print(f"  Fold {i + 1}: {s:.5f}")
    print(f"  -------------------------")
    print(f"  Mean:   {scores.mean():.5f}")
    print(f"  Std:    {scores.std():.5f}")
    print(f"  Time:   {elapsed:.1f}s")

    return scores


# ---------------------------------------------------------------------------
# 9. FEATURE CORRELATION FILTER
# ---------------------------------------------------------------------------
# Highly correlated features (>0.95) carry nearly identical information.
# Keeping both wastes training time and can destabilize tree-based
# models by randomly splitting credit between them, which makes feature
# importance unreliable. Dropping one of each correlated pair is a
# standard preprocessing step in tabular competitions.

def find_correlated_features(df, threshold=0.95):
    """
    Find pairs of numeric features with absolute correlation above threshold.

    Returns a set of column names recommended for dropping. For each
    correlated pair, the column that appears later in the DataFrame is
    selected for removal (arbitrary but consistent).

    Example:
        to_drop = tb.find_correlated_features(train, threshold=0.95)
        # Output:
        # [toolbox] 2 correlated pair(s) found (threshold=0.95):
        #   feature_A <-> feature_B: 0.9812
        #   feature_C <-> feature_D: 0.9634
        # [toolbox] Recommend dropping: {'feature_B', 'feature_D'}

        train.drop(columns=to_drop, inplace=True)  # Apply the recommendation
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()

    # Zero out the lower triangle and diagonal so each pair is counted once.
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    )

    to_drop = set()
    pairs = []

    for col in upper.columns:
        correlated = upper.index[upper[col] > threshold].tolist()
        for match in correlated:
            pairs.append((col, match, corr_matrix.loc[col, match]))
            to_drop.add(match)

    if pairs:
        print(f"[toolbox] {len(pairs)} correlated pair(s) found (threshold={threshold}):")
        for c1, c2, val in pairs[:10]:
            print(f"  {c1} <-> {c2}: {val:.4f}")
        if len(pairs) > 10:
            print(f"  ... and {len(pairs) - 10} more")
        print(f"[toolbox] Recommend dropping: {to_drop}")
    else:
        print(f"[toolbox] No features correlated above {threshold}")

    return to_drop


# ---------------------------------------------------------------------------
# 10. QUICK KAGGLE PATH FINDER
# ---------------------------------------------------------------------------
# When you add a dataset or utility script as input, Kaggle mounts it
# somewhere inside /kaggle/input/ or /kaggle/usr/lib/. The exact path
# depends on the source type and your username. This function walks
# through those directories and prints every file matching a keyword,
# saving you from manual directory exploration via !ls commands.

def find_input(pattern="", show_all=False):
    """
    Search all Kaggle input directories for files matching a keyword.

    If you cannot find your dataset or your import statement is failing,
    this function will tell you the exact path Kaggle used.

    Parameters:
        pattern  -- Keyword to search for (e.g., "train", "submission", "model")
        show_all -- If True and pattern is empty, lists everything available

    Example:
        tb.find_input("train")
        # Output:
        # [toolbox] Found 2 file(s) matching 'train':
        #   /kaggle/input/titanic/train.csv
        #   /kaggle/input/titanic/train_labels.csv

        tb.find_input("submission")
        # Output:
        # [toolbox] Found 1 file(s) matching 'submission':
        #   /kaggle/input/titanic/sample_submission.csv
    """
    search_dirs = ["/kaggle/input", "/kaggle/usr/lib", "/kaggle/working"]
    found = []

    for d in search_dirs:
        if not os.path.exists(d):
            continue
        for root, dirs, files in os.walk(d):
            for f in files:
                if show_all or pattern.lower() in f.lower():
                    full_path = os.path.join(root, f)
                    found.append(full_path)

    if found:
        print(f"[toolbox] Found {len(found)} file(s) matching '{pattern}':")
        for p in found[:30]:
            print(f"  {p}")
        if len(found) > 30:
            print(f"  ... and {len(found) - 30} more")
    else:
        print(f"[toolbox] No files found matching '{pattern}'")

    return found
