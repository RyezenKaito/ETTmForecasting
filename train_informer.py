"""
train_informer.py — Train the Informer model on ETTm1 (MS mode).

Decoder input follows the original Informer2020 paper:
  dec_inp = [label_len history ALL features | pred_len ZEROS ALL features]
  → No data leakage: the future portion is entirely zero-padded.

Usage:
    python train_informer.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config as cfg
from src.data.preprocessing import (
    load_and_split, add_time_features, fit_scaler, scale
)
from src.data.dataset import ETTDatasetInformer
from src.models.informer import Informer
from src.utils.metrics import metric
from src.utils.tools import EarlyStopping, plot_history


# ─────────────────────────────────────────────────────────────────────────────
# Build Informer datasets
# ─────────────────────────────────────────────────────────────────────────────

def build_informer_loaders(data_path, target_col, train_ratio, val_ratio,
                            seq_len, label_len, pred_len, batch_size):
    """
    Preprocessing pipeline (simplified — no STL, no Winsorization):
      1. Load & split  →  2. Drop cols  →  3. Add sin/cos  →  4. Scale
    Returns loaders + scaler + target column index.
    """
    train_df, val_df, test_df = load_and_split(data_path, train_ratio, val_ratio)

    # Drop correlated columns
    if hasattr(cfg, "DROP_COLS") and cfg.DROP_COLS:
        train_df = train_df.drop(columns=cfg.DROP_COLS, errors="ignore")
        val_df   = val_df.drop(columns=cfg.DROP_COLS, errors="ignore")
        test_df  = test_df.drop(columns=cfg.DROP_COLS, errors="ignore")

    # Time features (sin/cos — always appended as last 2 columns)
    train_df = add_time_features(train_df)
    val_df   = add_time_features(val_df)
    test_df  = add_time_features(test_df)

    # Scale (fit on train only)
    scaler    = fit_scaler(train_df)
    train_arr = scale(train_df, scaler)
    val_arr   = scale(val_df,   scaler)
    test_arr  = scale(test_df,  scaler)

    col_names  = list(train_df.columns)
    target_idx = col_names.index(target_col)

    print(f"  Columns ({len(col_names)}): {col_names}")
    print(f"  Target '{target_col}' at index {target_idx}")
    print(f"  Train {train_arr.shape} | Val {val_arr.shape} | Test {test_arr.shape}")

    # Build datasets (keep DatetimeIndex for time marks)
    train_ds = ETTDatasetInformer(train_arr, train_df.index, seq_len, label_len, pred_len)
    val_ds   = ETTDatasetInformer(val_arr,   val_df.index,   seq_len, label_len, pred_len)
    test_ds  = ETTDatasetInformer(test_arr,  test_df.index,  seq_len, label_len, pred_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader, scaler, target_idx


# ─────────────────────────────────────────────────────────────────────────────
# Process one batch  (MS mode — matches original Informer2020 paper)
# ─────────────────────────────────────────────────────────────────────────────

def process_batch(model, batch_x, batch_y, batch_x_mark, batch_y_mark,
                  pred_len, label_len, target_idx, device, padding=0):
    """
    Construct encoder/decoder inputs per the original Informer2020 paper:
      - Encoder:  batch_x  (all features)
      - Decoder:  [label_len steps ALL features | pred_len steps ZEROS ALL features]
      - Target:   batch_y[:, -pred_len:, target_idx]  (only OT)

    No data leakage: the future prediction window is entirely zero-padded.
    """
    batch_x      = batch_x.float().to(device)
    batch_y      = batch_y.float()
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)

    n_features = batch_y.shape[-1]  # = enc_in = dec_in = 7

    # Zero-padding for the prediction horizon (ALL features)
    if padding == 0:
        pred_pad = torch.zeros([batch_y.shape[0], pred_len, n_features]).float()
    else:
        pred_pad = torch.ones([batch_y.shape[0], pred_len, n_features]).float()

    # Decoder input: [label_len history | pred_len zeros]
    dec_inp = torch.cat([batch_y[:, :label_len, :], pred_pad], dim=1).float().to(device)

    # Forward pass
    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)  # (B, pred_len, c_out=1)

    # Ground truth: only the target column from the last pred_len steps
    ground_truth = batch_y[:, -pred_len:, target_idx:target_idx+1].to(device)

    return outputs, ground_truth   # both (B, pred_len, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(model, train_loader, val_loader, optimizer, criterion,
          pred_len, label_len, target_idx, device, epochs, patience, ckpt_path,
          scheduler=None):

    early_stop = EarlyStopping(patience=patience, verbose=True)
    history    = []

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for enc_x, dec_y, enc_mark, dec_mark in train_loader:
            optimizer.zero_grad()
            pred, true = process_batch(model, enc_x, dec_y, enc_mark, dec_mark,
                                       pred_len, label_len, target_idx, device)
            loss = criterion(pred, true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for enc_x, dec_y, enc_mark, dec_mark in val_loader:
                pred, true = process_batch(model, enc_x, dec_y, enc_mark, dec_mark,
                                           pred_len, label_len, target_idx, device)
                val_losses.append(criterion(pred.detach().cpu(), true.detach().cpu()).item())

        train_loss = float(np.mean(train_losses))
        val_loss   = float(np.mean(val_losses))
        lr_now     = optimizer.param_groups[0]["lr"]
        history.append((train_loss, val_loss))

        print(f"Epoch {epoch+1:03d} | Train {train_loss:.6f} | Val {val_loss:.6f} | LR {lr_now:.2e}")

        if scheduler is not None:
            scheduler.step(val_loss)

        early_stop(val_loss, model, ckpt_path)
        if early_stop.early_stop:
            print("Early stopping triggered.")
            break

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Test (evaluate on test set with original-scale metrics)
# ─────────────────────────────────────────────────────────────────────────────

def test(model, test_loader, scaler, target_idx, pred_len, label_len, device):
    """Run inference on the test set and compute metrics in original scale."""
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for enc_x, dec_y, enc_mark, dec_mark in test_loader:
            pred, true = process_batch(model, enc_x, dec_y, enc_mark, dec_mark,
                                       pred_len, label_len, target_idx, device)
            # Inverse transform to original scale
            pred_np = pred.squeeze(-1).cpu().numpy()
            true_np = true.squeeze(-1).cpu().numpy()
            pred_inv = pred_np * scaler.scale_[target_idx] + scaler.mean_[target_idx]
            true_inv = true_np * scaler.scale_[target_idx] + scaler.mean_[target_idx]
            preds.append(pred_inv)
            trues.append(true_inv)

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    results = metric(preds, trues)
    print(f"\n{'='*55}")
    print(f"  Informer Test Results (original scale)")
    print(f"{'='*55}")
    print(f"  MAE:  {results['MAE']:.4f}")
    print(f"  MSE:  {results['MSE']:.4f}")
    print(f"  RMSE: {results['RMSE']:.4f}")
    print(f"  MAPE: {results['MAPE']:.2f}%")
    print(f"{'='*55}")

    return preds, trues, results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Data
    print("Building Informer data loaders...")
    train_loader, val_loader, test_loader, scaler, target_idx = build_informer_loaders(
        cfg.DATA_PATH, cfg.TARGET_COL,
        cfg.TRAIN_RATIO, cfg.VAL_RATIO,
        cfg.SEQ_LEN, cfg.LABEL_LEN, cfg.PRED_LEN, cfg.INF_BATCH_SIZE
    )

    # 2. Model
    model = Informer(
        enc_in    = cfg.INF_ENC_IN,
        dec_in    = cfg.INF_DEC_IN,
        c_out     = cfg.INF_C_OUT,
        seq_len   = cfg.SEQ_LEN,
        label_len = cfg.LABEL_LEN,
        out_len   = cfg.PRED_LEN,
        factor    = cfg.INF_FACTOR,
        d_model   = cfg.INF_D_MODEL,
        n_heads   = cfg.INF_N_HEADS,
        e_layers  = cfg.INF_E_LAYERS,
        d_layers  = cfg.INF_D_LAYERS,
        d_ff      = cfg.INF_D_FF,
        dropout   = cfg.INF_DROPOUT,
        attn      = cfg.INF_ATTN,
        embed     = cfg.INF_EMBED,
        freq      = cfg.INF_FREQ,
        activation= cfg.INF_ACTIVATION,
        output_attention = cfg.INF_OUTPUT_ATTENTION,
        distil    = cfg.INF_DISTIL,
        mix       = cfg.INF_MIX,
        device    = device,
    ).float().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {train_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.INF_LR,
                                   weight_decay=cfg.INF_WEIGHT_DECAY)
    criterion = nn.HuberLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    # 3. Train
    os.makedirs(cfg.CKPT_DIR, exist_ok=True)
    history = train(model, train_loader, val_loader, optimizer, criterion,
                    cfg.PRED_LEN, cfg.LABEL_LEN, target_idx, device,
                    cfg.INF_EPOCHS, cfg.INF_PATIENCE, cfg.INF_CKPT,
                    scheduler=scheduler)

    # 4. Test (load best checkpoint)
    print("\nLoading best checkpoint for testing...")
    model.load_state_dict(torch.load(cfg.INF_CKPT, map_location=device, weights_only=True))
    preds, trues, results = test(model, test_loader, scaler, target_idx,
                                  cfg.PRED_LEN, cfg.LABEL_LEN, device)

    # 5. Plot training history
    os.makedirs(cfg.RESULT_DIR, exist_ok=True)
    plot_history(history, title="Informer Training Loss",
                 save_path=os.path.join(cfg.RESULT_DIR, "informer_loss.png"))

    print(f"\nDone! Best checkpoint saved to: {cfg.INF_CKPT}")


if __name__ == "__main__":
    main()
