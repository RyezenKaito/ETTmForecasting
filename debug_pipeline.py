import numpy as np
import pandas as pd
import math, time, os, copy
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import weight_norm
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
# ═══════════════════════════════════════════════════════════════
# DATA PIPELINE (verified correct)
# ═══════════════════════════════════════════════════════════════
torch.manual_seed(42); np.random.seed(42)

df = pd.read_csv('data\\ETTm1.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
for col in ['MUFL', 'MULL']:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

print(f'Columns: {df.columns.tolist()}')
print(f'Shape:   {df.shape}')


# --- 1.1.1 Thong ke mo ta co ban ---
print("=" * 60)
print("THONG KE MO TA DU LIEU")
print("=" * 60)
print(f"So luong mau: {len(df)}")
print(f"Khoang thoi gian: {df.index[0]} -> {df.index[-1]}")
print(f"Tan suat: 15 phut")
print(f"So luong features: {len(df.columns)}")
print()
print(df.describe().round(2))
# Tung bien duoc ve rieng de nhin ro hon
for col in df.columns:
    fig, ax = plt.subplots(figsize=(20, 4))
    ax.plot(df.index, df[col], linewidth=0.5, color='steelblue')
    ax.set_title(f'{col} theo thoi gian', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel(col)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
# Tung bien duoc ve rieng
for col in df.columns:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df[col], bins=60, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_title(f'Phan phoi cua {col}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Gia tri')
    ax.set_ylabel('Tan suat')
    ax.grid(True, alpha=0.3)

    # Them thong ke
    mean_val = df[col].mean()
    std_val = df[col].std()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.2f}')
    ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5, label=f'+1 Std = {mean_val+std_val:.2f}')
    ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5, label=f'-1 Std = {mean_val-std_val:.2f}')
    ax.legend()

    plt.tight_layout()
    plt.show()
# --- 1.1.4 Ma tran tuong quan ---
corr = df.corr()
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha='right')
ax.set_yticklabels(corr.columns)
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center',
                color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black', fontsize=9)
fig.colorbar(im)
ax.set_title('Ma tran tuong quan giua cac bien', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
# --- 1.1.5 Zoom vao target OT (1 thang dau) ---
one_month = df.iloc[:96*30]  # 30 ngay x 96 mau/ngay
fig, ax = plt.subplots(figsize=(20, 5))
ax.plot(one_month.index, one_month['OT'], linewidth=1, color='blue')
ax.set_title('OT - 1 thang dau tien (zoom)', fontsize=14, fontweight='bold')
ax.set_xlabel('Time')
ax.set_ylabel('OT (do C)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# --- 1.1.6 Kiem tra missing values ---
missing = df.isnull().sum()
print("Missing values:")
print(missing)
print(f"\nTong: {missing.sum()} missing values")
target_col   = 'OT'
n            = len(df)
train_size   = int(n * 0.6)
val_size     = int(n * 0.2)
test_size    = int(n * 0.2)
seq_len      = 336
label_len    = 48
pred_len     = 24
N_COVARIATE  = 7
batch_size   = 64

EPOCHS       = 100
PATIENCE     = 5
NOISE_STD    = 0.02
TREND_LAMBDA = 0.3
# def fix_freeze_blocks(df, window=6):
#     df = df.copy()
#     total_fixed = 0
#     for col in df.columns:
#         diff        = df[col].diff().abs()
#         is_frozen   = (diff == 0)
#         freeze_mask = is_frozen.rolling(window).sum() >= window

#         # Mở rộng mask về đầu block (rolling nhìn về cuối)
#         # Tìm điểm bắt đầu của mỗi freeze block
#         block_end   = freeze_mask & ~freeze_mask.shift(-1).fillna(False)
#         expanded    = freeze_mask.copy()
#         for end_ts in df.index[block_end]:
#             pos = df.index.get_loc(end_ts)
#             # Lùi lại để tìm điểm bắt đầu thật sự
#             start = pos
#             while start > 0 and diff.iloc[start] == 0:
#                 start -= 1
#             expanded.iloc[start+1:pos+1] = True

#         n_fixed = expanded.sum()
#         total_fixed += n_fixed
#         df.loc[expanded, col] = np.nan

#     df = df.interpolate(method='time')
#     # Fillna đầu/cuối nếu còn sót
#     df = df.ffill().bfill()
#     print(f'Fixed {total_fixed} frozen samples across all columns')
#     return df

# df = fix_freeze_blocks(df, window=6)
# Add time_sin,cos day_sin,cos feature
def add_time_features(dataframe):
    idx = dataframe.index
    t_intra = idx.hour * 4 + idx.minute // 15
    dataframe['time_sin'] = np.sin(2 * np.pi * t_intra / 96)
    dataframe['time_cos'] = np.cos(2 * np.pi * t_intra / 96)
    dataframe['day_sin']  = np.sin(2 * np.pi * idx.dayofweek / 7)
    dataframe['day_cos']  = np.cos(2 * np.pi * idx.dayofweek / 7)
    dataframe['week_sin'] = np.sin(2 * np.pi * idx.dayofweek)
    dataframe['week_cos'] = np.cos(2 * np.pi * idx.dayofweek)
    return dataframe

train_df = df.iloc[:train_size].copy()
val_df   = df.iloc[train_size:train_size + val_size].copy()
test_df  = df.iloc[train_size + val_size:].copy()
for _df in [train_df, val_df, test_df]:
    add_time_features(_df)
# =============================================
# STEP 1: Fit STL chỉ trên train
# =============================================
period = 96  # ETTm1: 15-phút × 96 = 1 ngày | ETTh1: dùng 24
stl = STL(train_df["OT"], period=period)
res = stl.fit()

train_df["trend"]    = res.trend.values
train_df["seasonal"] = res.seasonal.values
train_df["residual"] = res.resid.values

# =============================================
# STEP 2: Extract pattern cố định từ train
# =============================================

# Seasonal: lấy 1 chu kỳ đại diện (mean của từng vị trí trong chu kỳ)
seasonal_pattern = np.array([
    res.seasonal[i::period].mean()
    for i in range(period)
])  # shape: (period,) — 1 ngày điển hình

def apply_seasonal(df, pattern):
    """Tile seasonal pattern theo đúng vị trí giờ trong ngày"""
    n = len(df)
    # Căn theo vị trí trong ngày thực tế để không bị lệch pha
    start_offset = (df.index[0].hour * 4 + df.index[0].minute // 15) % period
    idx = [(start_offset + i) % period for i in range(n)]
    return np.array([pattern[i] for i in idx])

# Trend: rolling mean của OT (chỉ nhìn quá khứ → không leakage)
def apply_trend(df, window=96):
    return df["OT"].rolling(window=window, min_periods=1).mean().values

# =============================================
# STEP 3: Apply cho val và test
# =============================================
for split_df in [val_df, test_df]:
    split_df["trend"]    = apply_trend(split_df, window=period)
    split_df["seasonal"] = apply_seasonal(split_df, seasonal_pattern)
    split_df["residual"] = (split_df["OT"]
                            - split_df["trend"]
                            - split_df["seasonal"]).values
# Index của target
target_index = train_df.columns.get_loc(target_col)
# Số features
n_features   = len(train_df.columns)
# Scaler fit ONLY on train
scaler       = StandardScaler()
train_scaled = scaler.fit_transform(train_df.values)
val_scaled   = scaler.transform(val_df.values)
test_scaled  = scaler.transform(test_df.values)

print(f'target_index = {target_index}, n_features = {n_features}')
print(f'Scaler mean[OT]={scaler.mean_[target_index]:.2f}, scale[OT]={scaler.scale_[target_index]:.2f}')
print(f'Train scaled stats: mean={train_scaled[:,target_index].mean():.4f}, std={train_scaled[:,target_index].std():.4f}')

print('Data pipeline ready.')
