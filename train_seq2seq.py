"""
train_seq2seq.py — Train the Seq2SeqLSTM model on ETTm1.

Usage:
    python train_seq2seq.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config as cfg
from src.data.preprocessing import (
    load_and_split, add_time_features, fit_scaler, scale
)
from src.data.dataset import TimeSeriesDataset4Seq
from src.models.seq2seq import Seq2SeqLSTM
from src.utils.metrics import metric
from src.utils.tools import EarlyStopping, plot_history


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def inverse(x: np.ndarray, scaler, target_idx: int) -> np.ndarray:
    return x * scaler.scale_[target_idx] + scaler.mean_[target_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(model, train_loader, val_loader, optimizer, criterion,
          scheduler, device, scaler, target_idx, epochs, patience, ckpt_path):

    early_stop = EarlyStopping(patience=patience, verbose=True)
    history    = []

    for epoch in range(epochs):
        tf_ratio = max(0.6 * (0.98 ** epoch), 0.1)   # Slower teacher forcing decay
        model.train()
        train_losses = []

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            y_target = Y_batch[:, :, target_idx]

            # Construct leakage-free future features
            # We only provide the true future values of time_sin and time_cos (last 2 cols)
            future_feat = Y_batch[:, :, -2:]

            optimizer.zero_grad()
            output = model(X_batch, y=y_target,
                           future_features=future_feat,
                           teacher_forcing_ratio=tf_ratio)
            loss = criterion(output, y_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        # ── Validation ───────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_val, Y_val in val_loader:
                X_val = X_val.to(device)
                Y_val = Y_val.to(device)
                y_val_target = Y_val[:, :, target_idx]

                # Construct leakage-free future features (only sin, cos — last 2 cols)
                val_future = Y_val[:, :, -2:]

                out = model(X_val, y=None, future_features=val_future,
                            teacher_forcing_ratio=0.0)
                val_losses.append(criterion(out, y_val_target).item())

        train_loss = float(np.mean(train_losses))
        val_loss   = float(np.mean(val_losses))
        lr_now     = optimizer.param_groups[0]["lr"]
        history.append((train_loss, val_loss))

        print(f"Epoch {epoch+1:03d} | Train {train_loss:.6f} | Val {val_loss:.6f} | LR {lr_now:.2e}")
        scheduler.step(val_loss)
        early_stop(val_loss, model, ckpt_path)
        if early_stop.early_stop:
            print("Early stopping triggered.")
            break

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Test (evaluate on test set with original-scale metrics)
# ─────────────────────────────────────────────────────────────────────────────

def test(model, test_loader, scaler, target_idx, device):
    """Run inference on the test set and compute metrics in original scale."""
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            y_target = Y_batch[:, :, target_idx]
            future_feat = Y_batch[:, :, -2:]  # sin, cos

            out = model(X_batch, y=None, future_features=future_feat,
                        teacher_forcing_ratio=0.0)

            pred_inv = inverse(out.cpu().numpy(), scaler, target_idx)
            true_inv = inverse(y_target.cpu().numpy(), scaler, target_idx)
            preds.append(pred_inv)
            trues.append(true_inv)

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    results = metric(preds, trues)
    print(f"\n{'='*55}")
    print(f"  Seq2Seq Test Results (original scale)")
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

    # 1. Preprocessing (simplified: no STL, no Winsorization)
    print("Running preprocessing pipeline...")
    train_df, val_df, test_df = load_and_split(cfg.DATA_PATH, cfg.TRAIN_RATIO, cfg.VAL_RATIO)

    # Drop correlated columns
    if hasattr(cfg, "DROP_COLS") and cfg.DROP_COLS:
        train_df = train_df.drop(columns=cfg.DROP_COLS, errors="ignore")
        val_df   = val_df.drop(columns=cfg.DROP_COLS, errors="ignore")
        test_df  = test_df.drop(columns=cfg.DROP_COLS, errors="ignore")

    # Time features (sin/cos)
    train_df = add_time_features(train_df)
    val_df   = add_time_features(val_df)
    test_df  = add_time_features(test_df)

    # Scale (fit on train only)
    scaler       = fit_scaler(train_df)
    train_scaled = scale(train_df, scaler)
    val_scaled   = scale(val_df,   scaler)
    test_scaled  = scale(test_df,  scaler)

    col_names  = list(train_df.columns)
    target_idx = col_names.index(cfg.TARGET_COL)
    print(f"  Columns ({len(col_names)}): {col_names}")
    print(f"  Target '{cfg.TARGET_COL}' at index {target_idx}")
    print(f"  Train {train_scaled.shape} | Val {val_scaled.shape} | Test {test_scaled.shape}")

    # 2. Datasets & Loaders
    train_ds = TimeSeriesDataset4Seq(train_scaled, cfg.SEQ_LEN, cfg.PRED_LEN)
    val_ds   = TimeSeriesDataset4Seq(val_scaled,   cfg.SEQ_LEN, cfg.PRED_LEN)
    test_ds  = TimeSeriesDataset4Seq(test_scaled,  cfg.SEQ_LEN, cfg.PRED_LEN)

    train_loader = DataLoader(train_ds, batch_size=cfg.S2S_BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.S2S_BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.S2S_BATCH_SIZE, shuffle=False)

    # 3. Model
    model = Seq2SeqLSTM(
        input_dim   = train_scaled.shape[1],
        hidden_size = cfg.S2S_HIDDEN_SIZE,
        num_layers  = cfg.S2S_NUM_LAYERS,
        dropout     = cfg.S2S_DROPOUT,
        dec_in_dim  = cfg.S2S_DEC_IN_DIM,
        pred_len    = cfg.PRED_LEN,
        target_idx  = target_idx,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {train_params:,}")

    optimizer  = torch.optim.AdamW(model.parameters(), lr=cfg.S2S_LR,
                                   weight_decay=cfg.S2S_WEIGHT_DECAY)
    criterion  = nn.HuberLoss()
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    # 4. Train
    os.makedirs(cfg.CKPT_DIR, exist_ok=True)
    history = train(model, train_loader, val_loader, optimizer, criterion,
                    scheduler, device, scaler, target_idx,
                    cfg.S2S_EPOCHS, cfg.S2S_PATIENCE, cfg.S2S_CKPT)

    # 5. Test (load best checkpoint)
    print("\nLoading best checkpoint for testing...")
    model.load_state_dict(torch.load(cfg.S2S_CKPT, map_location=device, weights_only=True))
    preds, trues, results = test(model, test_loader, scaler, target_idx, device)

    # 6. Plot training history
    os.makedirs(cfg.RESULT_DIR, exist_ok=True)
    plot_history(history, title="Seq2Seq Training Loss",
                 save_path=os.path.join(cfg.RESULT_DIR, "seq2seq_loss.png"))

    print(f"\nDone! Best checkpoint saved to: {cfg.S2S_CKPT}")


if __name__ == "__main__":
    main()
