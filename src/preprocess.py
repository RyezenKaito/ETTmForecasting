"""
Data preprocessing pipeline – copy CHÍNH XÁC từ PhanPhungVu_PhamLeKhanhAn_Report35.ipynb.

Pipeline:
  1. Load ETTm1.csv, set index date
  2. Drop MUFL, MULL
  3. STL trên TOÀN BỘ df (đúng như Cell 16 notebook)
  4. Split 60/20/20
  5. add_time_features (time_sin/cos + day_sin/cos) – append cuối
  6. StandardScaler – fit CHỈ trên train
  7. Column order: HUFL, HULL, LUFL, LULL, OT, trend, seasonal, residual,
                   time_sin, time_cos, day_sin, day_cos
  8. target_index = 4 (OT)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL

# ---------- constants (khớp notebook Cell 18 + 25) ----------
SEQ_LEN   = 336
LABEL_LEN = 48
PRED_LEN  = 24
N_COV     = 4          # time_sin, time_cos, day_sin, day_cos
TARGET    = "OT"
PERIOD    = 96         # 15-min × 96 = 1 day


# ---------- helpers (khớp notebook Cell 22) ----------

def add_time_features(dataframe):
    """Thêm 4 time features vào cuối DataFrame (in-place + return)."""
    idx = dataframe.index
    t_intra = idx.hour * 4 + idx.minute // 15
    dataframe["time_sin"] = np.sin(2 * np.pi * t_intra / 96)
    dataframe["time_cos"] = np.cos(2 * np.pi * t_intra / 96)
    dataframe["day_sin"]  = np.sin(2 * np.pi * idx.dayofweek / 7)
    dataframe["day_cos"]  = np.cos(2 * np.pi * idx.dayofweek / 7)
    return dataframe


# ---------- main pipeline ----------

def build_pipeline(csv_path: str):
    """
    Replicate notebook pipeline EXACTLY.

    Returns:
        train_scaled, val_scaled, test_scaled  – np.ndarray
        scaler            – fitted StandardScaler
        target_index      – int  (= 4)
        n_features        – int  (= 12)
        train_df          – pd.DataFrame (column order reference)
    """
    # --- Cell 5: Load ---
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    # --- Cell 15: Drop MUFL, MULL ---
    for col in ["MUFL", "MULL"]:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # --- Cell 16: STL on FULL df (NOT train-only!) ---
    stl = STL(df[TARGET], period=PERIOD)
    res = stl.fit()
    df["trend"]    = res.trend
    df["seasonal"] = res.seasonal
    df["residual"] = res.resid

    # --- Cell 18: Split 60/20/20 ---
    n          = len(df)
    train_size = int(n * 0.6)
    val_size   = int(n * 0.2)

    train_df = df.iloc[:train_size].copy()
    val_df   = df.iloc[train_size: train_size + val_size].copy()
    test_df  = df.iloc[train_size + val_size:].copy()

    # --- Cell 22: Add time features (append to end) ---
    for _df in [train_df, val_df, test_df]:
        add_time_features(_df)

    # --- Cell 23/25: Verify column order ---
    # Expected: HUFL, HULL, LUFL, LULL, OT, trend, seasonal, residual,
    #           time_sin, time_cos, day_sin, day_cos
    target_index = list(train_df.columns).index(TARGET)
    n_features   = len(train_df.columns)

    # --- Cell 27: Scale ---
    scaler       = StandardScaler()
    train_scaled = scaler.fit_transform(train_df.values)
    val_scaled   = scaler.transform(val_df.values)
    test_scaled  = scaler.transform(test_df.values)

    return (
        train_scaled, val_scaled, test_scaled,
        scaler, target_index, n_features,
        train_df
    )
