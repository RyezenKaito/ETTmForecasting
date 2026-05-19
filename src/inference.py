"""
Inference helpers – load saved models and run predictions.
Khớp CHÍNH XÁC với PhanPhungVu_PhamLeKhanhAn_Report35.ipynb.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

from src.models import MODEL_CONFIGS
from src.metrics import inverse_target, calc_metrics

PRED_LEN = 24
N_COV    = 4


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len=336, label_len=48, pred_len=24):
        self.data      = data
        self.seq_len   = seq_len
        self.label_len = label_len
        self.pred_len  = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        s_end   = idx + self.seq_len
        r_begin = s_end - self.label_len
        r_end   = r_begin + self.label_len + self.pred_len
        seq_x = torch.tensor(self.data[idx:s_end],    dtype=torch.float32)
        seq_y = torch.tensor(self.data[r_begin:r_end], dtype=torch.float32)
        return seq_x, seq_y


def load_model(model_key, models_dir, n_features, pred_len, target_index, device):
    cfg   = MODEL_CONFIGS[model_key]
    wpath = os.path.join(models_dir, cfg["weight_file"])

    state = torch.load(wpath, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    kw = dict(cfg["kwargs"])
    kw.update(dict(input_dim=n_features, horizon=pred_len, pred_len=pred_len, target_index=target_index))

    model = cfg["cls"](**kw).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def _seq2seq_predict(model, Xb, Yb, pred_len):
    # Notebook Cell 38
    y_teacher = Yb[:, -pred_len:, model.target_index]
    out = model(Xb, y_teacher=y_teacher)
    return out.squeeze(-1) # Fix broadcasting bug when out is (N, 24, 1)


def _tcn_predict(model, Xb, Yb, pred_len):
    # Notebook Cell 47
    # FIX: best_tcn_v9_last_checkpoint.pth was actually trained on a pipeline 
    # where Time features came BEFORE STL features.
    # We dynamically reorder Xb and Yb here to Pipeline B order so it works perfectly.
    # Pipeline A: [0:5] orig, [5:8] STL, [8:12] Time
    # Pipeline B: [0:5] orig, [8:12] Time, [5:8] STL
    reorder = [0, 1, 2, 3, 4, 8, 9, 10, 11, 5, 6, 7]
    Xb_reordered = Xb[:, :, reorder]
    Yb_reordered = Yb[:, :, reorder]
    f_cov = Yb_reordered[:, -pred_len:, -N_COV:]
    return model(Xb_reordered, future_features=f_cov)


def _get_pred_fn(model_key):
    if model_key == "seq2seq":
        return _seq2seq_predict
    elif model_key == "tcn":
        return _tcn_predict
    else:
        raise ValueError(f"Unknown model_key: {model_key}")


def evaluate_model(model, model_key, loader, target_index, pred_len, scaler, device):
    pred_fn = _get_pred_fn(model_key)
    preds, trues = [], []

    model.eval()
    with torch.no_grad():
        for Xb, Yb in loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            y_true = Yb[:, -pred_len:, target_index]
            out    = pred_fn(model, Xb, Yb, pred_len)

            preds.append(inverse_target(out.cpu().numpy(),    scaler, target_index))
            trues.append(inverse_target(y_true.cpu().numpy(), scaler, target_index))

    preds_arr = np.concatenate(preds)
    trues_arr = np.concatenate(trues)
    return preds_arr, trues_arr, calc_metrics(preds_arr, trues_arr)


def predict_sample(model, model_key, test_scaled, sample_idx, pred_len,
                   target_index, scaler, device, seq_len=336, label_len=48):
    start = sample_idx
    s_end = start + seq_len
    r_begin = s_end - label_len
    r_end   = r_begin + label_len + pred_len

    Xb = torch.tensor(test_scaled[start:s_end],    dtype=torch.float32).unsqueeze(0).to(device)
    Yb = torch.tensor(test_scaled[r_begin:r_end],  dtype=torch.float32).unsqueeze(0).to(device)
    y_true_scaled = Yb[:, -pred_len:, target_index]

    pred_fn = _get_pred_fn(model_key)
    with torch.no_grad():
        out = pred_fn(model, Xb, Yb, pred_len)

    pred = inverse_target(out.cpu().numpy()[0],            scaler, target_index)
    true = inverse_target(y_true_scaled.cpu().numpy()[0],  scaler, target_index)
    return pred, true
