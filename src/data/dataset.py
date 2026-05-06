"""
src/data/dataset.py
PyTorch Dataset classes for both Seq2SeqLSTM and Informer.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Dataset for Seq2SeqLSTM
# ─────────────────────────────────────────────────────────────────────────────

class TimeSeriesDataset4Seq(Dataset):
    """
    Returns sliding windows for the Seq2Seq model.

    __getitem__ returns:
        X : Tensor (seq_len,  n_features)   — encoder input (past)
        Y : Tensor (pred_len, n_features)   — decoder target window (future,
                                               ALL features so the training loop
                                               can extract the target col and
                                               the future covariates separately)
    """

    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int):
        self.data     = data
        self.seq_len  = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Time-mark helpers for Informer
# ─────────────────────────────────────────────────────────────────────────────

def _time_features_from_index(index: pd.DatetimeIndex) -> np.ndarray:
    """
    Build a (N, 5) array of normalised time features used by Informer's timeF
    embedding with freq='t' (minutely):
      [month, day, weekday, hour, minute_slot]
    Values are divided by their maximum to sit in [-0.5, 0.5].
    freq='t' → freq_map['t'] = 5 columns in Informer's TimeFeatureEmbedding.
    """
    # minute slot: ETTm1 is 15-min → 4 slots per hour → [0,1,2,3]
    minute_slot = (index.minute.values // 15).astype(float)
    return np.stack([
        index.month.values           / 12.0   - 0.5,
        index.day.values             / 31.0   - 0.5,
        index.dayofweek.values       / 6.0    - 0.5,
        index.hour.values            / 23.0   - 0.5,
        minute_slot                  / 3.0    - 0.5,   # 4 slots → max index = 3
    ], axis=-1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset for Informer  (MS mode — multivariate → univariate OT)
# ─────────────────────────────────────────────────────────────────────────────

class ETTDatasetInformer(Dataset):
    """
    Dataset for Informer (MS mode).

    __getitem__ returns:
        enc_x      : (seq_len,              n_features)  — encoder input
        dec_y      : (label_len + pred_len, n_features)  — decoder input
                      [ label_len history | pred_len zeros ]
        enc_x_mark : (seq_len,              4)           — time features for enc
        dec_y_mark : (label_len + pred_len, 4)           — time features for dec

    The training loop later extracts batch_y[:, -pred_len:, target_col_idx]
    as the ground-truth for loss computation.
    """

    def __init__(self,
                 data: np.ndarray,
                 timestamps: pd.DatetimeIndex,
                 seq_len:   int,
                 label_len: int,
                 pred_len:  int):
        self.data      = data
        self.timestamps= timestamps
        self.seq_len   = seq_len
        self.label_len = label_len
        self.pred_len  = pred_len

        self.time_mark = _time_features_from_index(timestamps)

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        s_begin = idx
        s_end   = s_begin + self.seq_len
        r_begin = s_end   - self.label_len
        r_end   = r_begin + self.label_len + self.pred_len

        enc_x = self.data[s_begin : s_end]
        dec_y = self.data[r_begin : r_end]

        enc_x_mark = self.time_mark[s_begin : s_end]
        dec_y_mark = self.time_mark[r_begin : r_end]

        return (
            torch.tensor(enc_x,      dtype=torch.float32),
            torch.tensor(dec_y,      dtype=torch.float32),
            torch.tensor(enc_x_mark, dtype=torch.float32),
            torch.tensor(dec_y_mark, dtype=torch.float32),
        )
