"""
train_informer.py — Train the Informer model on ETTm1 (MS mode).

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
from src.data.preprocessing import build_pipeline, load_and_split
from src.data.dataset import ETTDatasetInformer
from src.models.informer import Informer
from src.utils.metrics import metric
from src.utils.tools import EarlyStopping, plot_history


# ─────────────────────────────────────────────────────────────────────────────
# Build Informer datasets (need the original DatetimeIndex for time marks)
# ─────────────────────────────────────────────────────────────────────────────

def build_informer_loaders(data_path, target_col, stl_period, train_ratio, val_ratio,
                            seq_len, label_len, pred_len, batch_size):
    """
    Re-runs the preprocessing pipeline but also keeps the DatetimeIndex
    needed to build time marks for the Informer embedding.
    """
    from src.data.preprocessing import (
        load_and_split, fit_stl, apply_stl_features, add_time_features,
        compute_clip_bounds, winsorize, fit_scaler, scale
    )

    train_df, val_df, test_df = load_and_split(data_path, train_ratio, val_ratio)

    if hasattr(cfg, "DROP_COLS") and cfg.DROP_COLS:
        train_df = train_df.drop(columns=cfg.DROP_COLS, errors="ignore")
        val_df   = val_df.drop(columns=cfg.DROP_COLS, errors="ignore")
        test_df  = test_df.drop(columns=cfg.DROP_COLS, errors="ignore")

    _, seasonal_pattern = fit_stl(train_df, target_col, stl_period)

    train_df = apply_stl_features(train_df, seasonal_pattern, target_col, stl_period)
    val_df   = apply_stl_features(val_df,   seasonal_pattern, target_col, stl_period)
    test_df  = apply_stl_features(test_df,  seasonal_pattern, target_col, stl_period)

    train_df = add_time_features(train_df)
    val_df   = add_time_features(val_df)
    test_df  = add_time_features(test_df)

    bounds    = compute_clip_bounds(train_df)
    train_df  = winsorize(train_df, bounds)
    val_df    = winsorize(val_df,   bounds)
    test_df   = winsorize(test_df,  bounds)

    scaler      = fit_scaler(train_df)
    train_arr   = scale(train_df, scaler)
    val_arr     = scale(val_df,   scaler)
    test_arr    = scale(test_df,  scaler)

    col_names  = list(train_df.columns)
    target_idx = col_names.index(target_col)

    # Build datasets (keep DatetimeIndex for time marks)
    train_ds = ETTDatasetInformer(train_arr, train_df.index, seq_len, label_len, pred_len)
    val_ds   = ETTDatasetInformer(val_arr,   val_df.index,   seq_len, label_len, pred_len)
    test_ds  = ETTDatasetInformer(test_arr,  test_df.index,  seq_len, label_len, pred_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader, scaler, target_idx


# ─────────────────────────────────────────────────────────────────────────────
# Process one batch (MS mode: predict only the target column)
# ─────────────────────────────────────────────────────────────────────────────

def process_batch(model, batch_x, batch_y, batch_x_mark, batch_y_mark,
                  pred_len, label_len, target_idx, device, padding=0):
    batch_x      = batch_x.float().to(device)
    batch_y      = batch_y.float()
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)

    # Subset features for Decoder: [OT, time_sin, time_cos]
    # This matches DEC_IN_DIM = 3
    # OT is at target_idx, time_sin/cos are the last 2 columns (-2, -1)
    batch_y_dec = batch_y[:, :, [target_idx, -2, -1]]

    # Decoder input: [label_len history | zero padding for pred_len]
    if padding == 0:
        pred_pad = torch.zeros([batch_y_dec.shape[0], pred_len, 3]).float()
    else:
        pred_pad = torch.ones([batch_y_dec.shape[0], pred_len, 3]).float()

    # Overwrite the known deterministic features in the padded future (sin=index 1, cos=index 2 in the 3-feature slice)
    pred_pad[:, :, 1:3] = batch_y_dec[:, -pred_len:, 1:3]

    dec_inp = torch.cat([batch_y_dec[:, :label_len, :], pred_pad], dim=1).float().to(device)

    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)  # (B, pred_len, c_out)

    # MS mode: take only the target column from the last pred_len steps of batch_y
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
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Data
    print("Building Informer data loaders...")
    train_loader, val_loader, test_loader, scaler, target_idx = build_informer_loaders(
        cfg.DATA_PATH, cfg.TARGET_COL, cfg.STL_PERIOD,
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

    # 4. Plot
    os.makedirs(cfg.RESULT_DIR, exist_ok=True)
    plot_history(history, title="Informer Training Loss",
                 save_path=os.path.join(cfg.RESULT_DIR, "informer_loss.png"))

    print(f"\nDone! Best checkpoint saved to: {cfg.INF_CKPT}")


if __name__ == "__main__":
    main()
