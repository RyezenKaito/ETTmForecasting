"""
Data preprocessing pipeline – theo file PhanPhungVu (pipeline thật).
Bước:
  1. Load ETTm1.csv, set index date
  2. Drop MUFL, MULL (nếu tồn tại)
  3. Split 60/20/20
  4. add_time_features (time_sin/cos + day_sin/cos)
  5. STL – fit CHỈ trên train, apply seasonal pattern cho val/test
  6. StandardScaler – fit CHỈ trên train
  7. Trả về train/val/test scaled arrays + meta
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL

# ---------- constants ----------
SEQ_LEN   = 336
LABEL_LEN = 48
PRED_LEN  = 24
N_COV     = 4          # time_sin, time_cos, day_sin, day_cos
TARGET    = "OT"
PERIOD    = 96         # 15-min × 96 = 1 day


# ---------- helpers ----------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    t_intra = idx.hour * 4 + idx.minute // 15
    df = df.copy()
    df["time_sin"] = np.sin(2 * np.pi * t_intra / 96)
    df["time_cos"] = np.cos(2 * np.pi * t_intra / 96)
    df["day_sin"]  = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["day_cos"]  = np.cos(2 * np.pi * idx.dayofweek / 7)
    return df


def _apply_seasonal(df: pd.DataFrame, pattern: np.ndarray) -> np.ndarray:
    n = len(df)
    start_offset = (df.index[0].hour * 4 + df.index[0].minute // 15) % PERIOD
    idx = [(start_offset + i) % PERIOD for i in range(n)]
    return np.array([pattern[i] for i in idx])


def _apply_trend(df: pd.DataFrame, window: int = PERIOD) -> np.ndarray:
    return df[TARGET].rolling(window=window, min_periods=1).mean().values


# ---------- main pipeline ----------

def build_pipeline(csv_path: str):
    """
    Returns:
        train_scaled, val_scaled, test_scaled  – np.ndarray
        scaler            – fitted StandardScaler
        target_index      – int
        n_features        – int
        seasonal_pattern  – np.ndarray  (shape: (PERIOD,))
        train_df          – pd.DataFrame (column order reference)
    """
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    for col in ["MUFL", "MULL"]:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    n          = len(df)
    train_size = int(n * 0.6)
    val_size   = int(n * 0.2)

    train_df = df.iloc[:train_size].copy()
    val_df   = df.iloc[train_size: train_size + val_size].copy()
    test_df  = df.iloc[train_size + val_size:].copy()

    # time features
    for split in [train_df, val_df, test_df]:
        split = add_time_features(split)
    train_df = add_time_features(train_df)
    val_df   = add_time_features(val_df)
    test_df  = add_time_features(test_df)

    # STL – fit only on train
    stl = STL(train_df[TARGET], period=PERIOD)
    res = stl.fit()
    train_df["trend"]    = res.trend.values
    train_df["seasonal"] = res.seasonal.values
    train_df["residual"] = res.resid.values

    seasonal_pattern = np.array([
        res.seasonal[i::PERIOD].mean() for i in range(PERIOD)
    ])

    for split_df in [val_df, test_df]:
        split_df["trend"]    = _apply_trend(split_df, window=PERIOD)
        split_df["seasonal"] = _apply_seasonal(split_df, seasonal_pattern)
        split_df["residual"] = (
            split_df[TARGET] - split_df["trend"] - split_df["seasonal"]
        ).values



    target_index = list(train_df.columns).index(TARGET)
    n_features   = len(train_df.columns)

    # scale
    scaler       = StandardScaler()
    train_scaled = scaler.fit_transform(train_df.values)
    val_scaled   = scaler.transform(val_df.values)
    test_scaled  = scaler.transform(test_df.values)

    return (
        train_scaled, val_scaled, test_scaled,
        scaler, target_index, n_features,
        seasonal_pattern, train_df
    )
