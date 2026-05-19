"""
Data preprocessing pipeline – theo file PhanPhungVu.
Bước:
  1. Load ETTm1.csv, set index date
  2. Drop MUFL, MULL (nếu tồn tại)
  3. STL – fit trên toàn bộ df (đúng như notebook gốc)
  4. Split 60/20/20
  5. add_time_features (time_sin/cos + day_sin/cos) → luôn ở cuối
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
TIME_COLS = ["time_sin", "time_cos", "day_sin", "day_cos"]


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


def _reorder_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep time features as the last 4 columns for TCN covariates."""
    missing = [c for c in TIME_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing time features: {missing}")
    cols = [c for c in df.columns if c not in TIME_COLS] + TIME_COLS
    return df[cols]


# ---------- main pipeline ----------

def build_pipeline(csv_path: str):
    """
    Returns:
        train_scaled, val_scaled, test_scaled  – np.ndarray
        scaler            – fitted StandardScaler
        target_index      – int
        n_features        – int
        seasonal_pattern  – None (giữ chữ ký hàm)
        train_df          – pd.DataFrame (column order reference)
    """
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    for col in ["MUFL", "MULL"]:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # STL – fit on full df (match notebook)
    stl = STL(df[TARGET], period=PERIOD)
    res = stl.fit()
    df["trend"]    = res.trend.values
    df["seasonal"] = res.seasonal.values
    df["residual"] = res.resid.values

    n          = len(df)
    train_size = int(n * 0.6)
    val_size   = int(n * 0.2)

    train_df = df.iloc[:train_size].copy()
    val_df   = df.iloc[train_size: train_size + val_size].copy()
    test_df  = df.iloc[train_size + val_size:].copy()

    # time features (append last)
    splits = [train_df, val_df, test_df]
    splits = [add_time_features(split) for split in splits]
    splits = [_reorder_time_features(split) for split in splits]
    train_df, val_df, test_df = splits


    target_index = list(train_df.columns).index(TARGET)
    n_features   = len(train_df.columns)

    # scale
    scaler       = StandardScaler()
    train_scaled = scaler.fit_transform(train_df.values)
    val_scaled   = scaler.transform(val_df.values)
    test_scaled  = scaler.transform(test_df.values)

    seasonal_pattern = None
    return (
        train_scaled, val_scaled, test_scaled,
        scaler, target_index, n_features,
        seasonal_pattern, train_df
    )
