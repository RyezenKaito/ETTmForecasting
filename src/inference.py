"""
Inference helpers – load a saved model and run predictions.
Auto-detects input_dim from checkpoint to handle different training configs.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.models import MODEL_CONFIGS
from src.metrics import inverse_target, calc_metrics

PRED_LEN = 24
N_COV    = 4


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len=336, label_len=48, pred_len=24):
        self.data, self.seq_len = data, seq_len
        self.label_len, self.pred_len = label_len, pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        s_end   = idx + self.seq_len
        r_begin = s_end - self.label_len
        r_end   = r_begin + self.label_len + self.pred_len
        return (torch.tensor(self.data[idx:s_end],    dtype=torch.float32),
                torch.tensor(self.data[r_begin:r_end], dtype=torch.float32))


def _detect_input_dim(state_dict, model_key):
    """Detect true input_dim from checkpoint weights."""
    if model_key in ("seq2seq", "attention"):
        # BiLSTM / LSTM: weight_ih_l0 has shape [4*hidden, input_dim]
        for k, v in state_dict.items():
            if "encoder" in k and "weight_ih_l0" in k and "reverse" not in k:
                return v.shape[1]
    if model_key == "tcn":
        # weight_norm splits weight into weight_v [out, in, k] and weight_g [out,1,1]
        # We MUST use weight_v (not weight_g which gives dim 1)
        for k, v in state_dict.items():
            if "conv1" in k and "weight_v" in k and v.dim() == 3:
                return v.shape[1]
        # fallback: LayerNorm on input has shape [input_dim]
        for k, v in state_dict.items():
            if "norm1.weight" in k and v.dim() == 1:
                return int(v.shape[0])
    return None


def _pad_to(tensor, model_dim):
    """Zero-pad last dimension if model expects more features than data has."""
    data_dim = tensor.shape[-1]
    if model_dim <= data_dim:
        return tensor
    diff = model_dim - data_dim
    pad  = torch.zeros(*tensor.shape[:-1], diff, device=tensor.device)
    return torch.cat([tensor, pad], dim=-1)


def load_model(model_key, models_dir, n_features, pred_len, target_index, device):
    cfg   = MODEL_CONFIGS[model_key]
    wpath = os.path.join(models_dir, cfg["weight_file"])
    state = torch.load(wpath, map_location=device, weights_only=True)

    detected   = _detect_input_dim(state, model_key)
    actual_dim = detected if detected else n_features
    print(f"  [{model_key}] checkpoint input_dim detected={detected}, using={actual_dim}")

    kw = dict(cfg["kwargs"])
    if model_key == "tcn":
        kw.update(dict(input_dim=actual_dim, horizon=pred_len,
                       target_index=target_index))
    else:
        kw.update(dict(input_dim=actual_dim, pred_len=pred_len,
                       target_index=target_index))

    model = cfg["cls"](**kw).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, actual_dim


def evaluate_model(model, model_key, loader, target_index, pred_len,
                   scaler, device, model_dim=None):
    preds, trues = [], []
    with torch.no_grad():
        for Xb, Yb in loader:
            Xb, Yb    = Xb.to(device), Yb.to(device)
            y_true    = Yb[:, -pred_len:, target_index]
            Xb_in     = _pad_to(Xb, model_dim) if model_dim else Xb
            Yb_in     = _pad_to(Yb, model_dim) if model_dim else Yb

            if model_key == "tcn":
                f_cov = Yb[:, -pred_len:, -4:]     # TCN expects 4 covariates
                out   = model(Xb_in, future_features=f_cov)
            else:
                out   = model(Xb_in)

            preds.append(inverse_target(out.cpu().numpy(),    scaler, target_index))
            trues.append(inverse_target(y_true.cpu().numpy(), scaler, target_index))

    preds_arr = np.concatenate(preds)
    trues_arr = np.concatenate(trues)



    return preds_arr, trues_arr, calc_metrics(preds_arr, trues_arr)


def predict_sample(model, model_key, test_scaled, sample_idx, pred_len,
                   target_index, scaler, device, seq_len=336, model_dim=None):
    start, end = sample_idx, sample_idx + seq_len
    Xb = torch.tensor(test_scaled[start:end],      dtype=torch.float32).unsqueeze(0).to(device)
    Yb = torch.tensor(test_scaled[end:end+pred_len], dtype=torch.float32).unsqueeze(0).to(device)
    y_true_scaled = Yb[:, :, target_index]

    Xb_in = _pad_to(Xb, model_dim) if model_dim else Xb
    Yb_in = _pad_to(Yb, model_dim) if model_dim else Yb

    with torch.no_grad():
        if model_key == "tcn":
            out = model(Xb_in, future_features=Yb[:, :, -4:])
        else:
            out = model(Xb_in)

    pred = inverse_target(out.cpu().numpy()[0],            scaler, target_index)
    true = inverse_target(y_true_scaled.cpu().numpy()[0],  scaler, target_index)
    

    
    return pred, true
