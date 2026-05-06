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
from src.data.preprocessing import build_pipeline
from src.data.dataset import TimeSeriesDataset4Seq
from src.models.seq2seq import Seq2SeqLSTM
from src.utils.metrics import metric
from src.utils.tools import EarlyStopping, plot_history


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def inverse(x: np.ndarray, scaler, target_idx: int) -> np.ndarray:
    return x * scaler.scale_[target_idx] + scaler.mean_[target_idx]


def mse_real(pred, true, scaler, target_idx):
    p = inverse(pred, scaler, target_idx)
    t = inverse(true, scaler, target_idx)
    return float(np.mean((p - t) ** 2))


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

            train_losses.append(mse_real(
                output.detach().cpu().numpy(),
                y_target.detach().cpu().numpy(),
                scaler, target_idx
            ))

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
                val_losses.append(mse_real(
                    out.detach().cpu().numpy(),
                    y_val_target.detach().cpu().numpy(),
                    scaler, target_idx
                ))

        train_loss = float(np.mean(train_losses))
        val_loss   = float(np.mean(val_losses))
        lr_now     = optimizer.param_groups[0]["lr"]
        history.append((train_loss, val_loss))

        print(f"Epoch {epoch+1:03d} | Train {train_loss:.4f} | Val {val_loss:.4f} | LR {lr_now:.6f}")
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

    # 1. Preprocessing
    print("Running preprocessing pipeline...")
    train_scaled, val_scaled, test_scaled, scaler, target_idx, col_names = build_pipeline(
        cfg.DATA_PATH, cfg.TARGET_COL, cfg.STL_PERIOD,
        cfg.TRAIN_RATIO, cfg.VAL_RATIO, drop_cols=cfg.DROP_COLS
    )
    print(f"  Columns ({len(col_names)}): {col_names}")
    print(f"  Target index: {target_idx}")
    print(f"  Train {train_scaled.shape} | Val {val_scaled.shape} | Test {test_scaled.shape}")

    # 2. Datasets & Loaders
    train_ds = TimeSeriesDataset4Seq(train_scaled, cfg.SEQ_LEN, cfg.PRED_LEN)
    val_ds   = TimeSeriesDataset4Seq(val_scaled,   cfg.SEQ_LEN, cfg.PRED_LEN)

    train_loader = DataLoader(train_ds, batch_size=cfg.S2S_BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.S2S_BATCH_SIZE, shuffle=False)

    # 3. Model
    model = Seq2SeqLSTM(
        input_dim   = train_scaled.shape[1],
        hidden_size = cfg.S2S_HIDDEN_SIZE,
        num_layers  = cfg.S2S_NUM_LAYERS,
        dropout     = cfg.S2S_DROPOUT,
        dec_in_dim  = cfg.DEC_IN_DIM,
        pred_len    = cfg.PRED_LEN,
        target_idx  = target_idx,
    ).to(device)

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

    # 5. Plot
    os.makedirs(cfg.RESULT_DIR, exist_ok=True)
    plot_history(history, title="Seq2Seq Training Loss",
                 save_path=os.path.join(cfg.RESULT_DIR, "seq2seq_loss.png"))

    print(f"\nDone! Best checkpoint saved to: {cfg.S2S_CKPT}")


if __name__ == "__main__":
    main()
