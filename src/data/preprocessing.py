"""
src/data/preprocessing.py
Handles all data loading, splitting, feature engineering, and scaling.
All fitting (STL, StandardScaler, IQR bounds) is done ONLY on the training set
to prevent data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL


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
# 2. STL Decomposition  (fit on train only, apply to val/test via rolling)
# ─────────────────────────────────────────────────────────────────────────────

def fit_stl(train_df: pd.DataFrame, target_col: str = "OT", period: int = 96):
    """
    Run STL on the training OT series.
    Returns (stl_result, seasonal_pattern).
      - seasonal_pattern : shape (period,), one representative daily cycle.
    """
    res = STL(train_df[target_col], period=period, robust=True).fit()

    # Extract one representative seasonal cycle (mean of each intra-day position)
    seasonal_pattern = np.array([
        res.seasonal[i::period].mean() for i in range(period)
    ])

    return res, seasonal_pattern


def apply_stl_features(df: pd.DataFrame, seasonal_pattern: np.ndarray,
                       target_col: str = "OT", period: int = 96):
    """
    Add trend / seasonal / residual columns to df.
    - trend    : rolling mean of OT (past-only → no leakage)
    - seasonal : tile from the pattern extracted from train
    - residual : OT - trend - seasonal
    """
    df = df.copy()

    # Trend: causal rolling mean
    df["trend"] = df[target_col].rolling(window=period, min_periods=1).mean().values

    # Seasonal: tile pattern aligned to time-of-day
    n = len(df)
    start_offset = (df.index[0].hour * 4 + df.index[0].minute // 15) % period
    idx = [(start_offset + i) % period for i in range(n)]
    df["seasonal"] = np.array([seasonal_pattern[i] for i in idx])

    # Residual
    df["residual"] = (df[target_col] - df["trend"] - df["seasonal"]).values

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Time Features  (sin/cos encoding, no leakage — deterministic)
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
# 4. Winsorization  (IQR bounds fitted on train only)
# ─────────────────────────────────────────────────────────────────────────────

def compute_clip_bounds(train_df: pd.DataFrame):
    """Compute IQR-based clipping bounds from the training set."""
    bounds = {}
    for col in train_df.columns:
        Q1  = train_df[col].quantile(0.25)
        Q3  = train_df[col].quantile(0.75)
        IQR = Q3 - Q1
        bounds[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    return bounds


def winsorize(df: pd.DataFrame, bounds: dict):
    """Clip each column of df to the precomputed bounds."""
    df = df.copy()
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Scaling  (StandardScaler fitted on train only)
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
# 6. Inverse Transform  (scalar version for a single target column)
# ─────────────────────────────────────────────────────────────────────────────

def inverse_transform_target(x: np.ndarray, scaler: StandardScaler, target_idx: int) -> np.ndarray:
    """
    Reverse StandardScaler for the target column only.
    x : numpy array of scaled target values, shape (N,) or (N, T)
    """
    return x * scaler.scale_[target_idx] + scaler.mean_[target_idx]


# ─────────────────────────────────────────────────────────────────────────────
# 7. Convenience: run the full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline(data_path: str, target_col: str = "OT", period: int = 96,
                   train_ratio: float = 0.6, val_ratio: float = 0.2,
                   drop_cols: list = None):
    """
    End-to-end preprocessing. Returns:
        train_scaled, val_scaled, test_scaled : np.ndarray  (N, n_features)
        scaler       : fitted StandardScaler
        target_idx   : int  (column index of target in the scaled arrays)
        col_names    : list[str]
    """
    # 1. Load & split
    train_df, val_df, test_df = load_and_split(data_path, train_ratio, val_ratio)

    # 1.5 Drop highly-correlated columns (e.g. MUFL, MULL)
    if drop_cols:
        train_df = train_df.drop(columns=drop_cols, errors="ignore")
        val_df   = val_df.drop(columns=drop_cols, errors="ignore")
        test_df  = test_df.drop(columns=drop_cols, errors="ignore")

    # 2. STL on train
    _, seasonal_pattern = fit_stl(train_df, target_col, period)
    train_df = apply_stl_features(train_df, seasonal_pattern, target_col, period)
    val_df   = apply_stl_features(val_df,   seasonal_pattern, target_col, period)
    test_df  = apply_stl_features(test_df,  seasonal_pattern, target_col, period)

    # 3. Time features (always appended last → sin/cos are the last 2 columns)
    train_df = add_time_features(train_df)
    val_df   = add_time_features(val_df)
    test_df  = add_time_features(test_df)

    # 4. Winsorize (bounds from train)
    bounds  = compute_clip_bounds(train_df)
    train_df = winsorize(train_df, bounds)
    val_df   = winsorize(val_df,   bounds)
    test_df  = winsorize(test_df,  bounds)

    # 5. Scale (fit on train)
    scaler      = fit_scaler(train_df)
    train_scaled = scale(train_df, scaler)
    val_scaled   = scale(val_df,   scaler)
    test_scaled  = scale(test_df,  scaler)

    col_names  = list(train_df.columns)
    target_idx = col_names.index(target_col)

    return train_scaled, val_scaled, test_scaled, scaler, target_idx, col_names

