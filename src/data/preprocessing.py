"""
src/data/preprocessing.py
Handles all data loading, splitting, feature engineering, and scaling.
All fitting (StandardScaler) is done ONLY on the training set
to prevent data leakage.

Pipeline (simplified — no STL, no Winsorization):
  1. load_and_split()      → train_df, val_df, test_df
  2. drop MUFL, MULL       → 5 cols: HUFL, HULL, LUFL, LULL, OT
  3. add_time_features()   → +2 cols: time_sin, time_cos  → total 7
  4. fit_scaler() on train → StandardScaler
  5. scale() all splits    → numpy arrays
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load and Split
# ─────────────────────────────────────────────────────────────────────────────

def load_and_split(data_path: str, train_ratio: float = 0.6, val_ratio: float = 0.2):
    """
    Load ETTm1.csv and split chronologically into train / val / test.
    Returns (train_df, val_df, test_df) — all with DatetimeIndex.
    """
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    n = len(df)
    train_end = int(n * train_ratio)
    val_end   = train_end + int(n * val_ratio)

    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()
    test_df  = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Time Features  (sin/cos encoding, no leakage — deterministic)
# ─────────────────────────────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame):
    """
    Add cyclical time-of-day features (sin/cos of position within the day).
    96 slots per day for 15-min ETTm1 data.
    """
    df = df.copy()
    t = df.index.hour * 4 + df.index.minute // 15
    df["time_sin"] = np.sin(2 * np.pi * t / 96)
    df["time_cos"] = np.cos(2 * np.pi * t / 96)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Scaling  (StandardScaler fitted on train only)
# ─────────────────────────────────────────────────────────────────────────────

def fit_scaler(train_df: pd.DataFrame):
    """Fit a StandardScaler on the training DataFrame and return it."""
    scaler = StandardScaler()
    scaler.fit(train_df.values)
    return scaler


def scale(df: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """Transform df using an already-fitted scaler."""
    return scaler.transform(df.values)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Inverse Transform  (scalar version for a single target column)
# ─────────────────────────────────────────────────────────────────────────────

def inverse_transform_target(x: np.ndarray, scaler: StandardScaler, target_idx: int) -> np.ndarray:
    """
    Reverse StandardScaler for the target column only.
    x : numpy array of scaled target values, shape (N,) or (N, T)
    """
    return x * scaler.scale_[target_idx] + scaler.mean_[target_idx]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Convenience: run the full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline(data_path: str, target_col: str = "OT",
                   train_ratio: float = 0.6, val_ratio: float = 0.2,
                   drop_cols: list = None):
    """
    End-to-end preprocessing. Returns:
        train_scaled, val_scaled, test_scaled : np.ndarray  (N, n_features)
        scaler       : fitted StandardScaler
        target_idx   : int  (column index of target in the scaled arrays)
        col_names    : list[str]

    Pipeline:
        1. Load & split chronologically
        2. Drop correlated columns (MUFL, MULL)
        3. Add time features (sin/cos)
        4. Fit StandardScaler on train
        5. Scale all splits
    """
    # 1. Load & split
    train_df, val_df, test_df = load_and_split(data_path, train_ratio, val_ratio)

    # 2. Drop highly-correlated columns (e.g. MUFL, MULL)
    if drop_cols:
        train_df = train_df.drop(columns=drop_cols, errors="ignore")
        val_df   = val_df.drop(columns=drop_cols, errors="ignore")
        test_df  = test_df.drop(columns=drop_cols, errors="ignore")

    # 3. Time features (always appended last → sin/cos are the last 2 columns)
    train_df = add_time_features(train_df)
    val_df   = add_time_features(val_df)
    test_df  = add_time_features(test_df)

    # 4. Scale (fit on train)
    scaler       = fit_scaler(train_df)
    train_scaled = scale(train_df, scaler)
    val_scaled   = scale(val_df,   scaler)
    test_scaled  = scale(test_df,  scaler)

    col_names  = list(train_df.columns)
    target_idx = col_names.index(target_col)

    return train_scaled, val_scaled, test_scaled, scaler, target_idx, col_names
