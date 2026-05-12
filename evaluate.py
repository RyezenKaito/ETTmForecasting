"""
evaluate.py — Load saved checkpoints for both models and compare on the test set.

Usage:
    python evaluate.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import config as cfg
from src.data.preprocessing import (
    load_and_split, add_time_features, fit_scaler, scale
)
from src.data.dataset import TimeSeriesDataset4Seq, ETTDatasetInformer
from src.models.seq2seq import Seq2SeqLSTM
from src.models.informer import Informer
from src.utils.metrics import metric


# ─────────────────────────────────────────────────────────────────────────────
# Inverse transform helper
# ─────────────────────────────────────────────────────────────────────────────

def inverse(x, scaler, target_idx):
    return x * scaler.scale_[target_idx] + scaler.mean_[target_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate Seq2Seq
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_seq2seq(test_scaled, scaler, target_idx, device):
    test_ds     = TimeSeriesDataset4Seq(test_scaled, cfg.SEQ_LEN, cfg.PRED_LEN)
    test_loader = DataLoader(test_ds, batch_size=cfg.S2S_BATCH_SIZE, shuffle=False)

    model = Seq2SeqLSTM(
        input_dim   = test_scaled.shape[1],
        hidden_size = cfg.S2S_HIDDEN_SIZE,
        num_layers  = cfg.S2S_NUM_LAYERS,
        dropout     = cfg.S2S_DROPOUT,
        dec_in_dim  = cfg.S2S_DEC_IN_DIM,
        pred_len    = cfg.PRED_LEN,
        target_idx  = target_idx,
    ).to(device)
    model.load_state_dict(torch.load(cfg.S2S_CKPT, map_location=device, weights_only=True))
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            y_target = Y_batch[:, :, target_idx]

            # Construct leakage-free future features (sin, cos — last 2 cols)
            future_feat = Y_batch[:, :, -2:]

            out = model(X_batch, y=None, future_features=future_feat,
                        teacher_forcing_ratio=0.0)
            preds.append(inverse(out.cpu().numpy(), scaler, target_idx))
            trues.append(inverse(y_target.cpu().numpy(), scaler, target_idx))

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return preds, trues


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate Informer  (decoder receives ALL features — per original paper)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_informer(test_scaled, test_index, scaler, target_idx, device):
    test_ds = ETTDatasetInformer(test_scaled, test_index,
                                  cfg.SEQ_LEN, cfg.LABEL_LEN, cfg.PRED_LEN)
    test_loader = DataLoader(test_ds, batch_size=cfg.INF_BATCH_SIZE,
                             shuffle=False, drop_last=True)

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
    model.load_state_dict(torch.load(cfg.INF_CKPT, map_location=device, weights_only=True))
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for enc_x, dec_y, enc_mark, dec_mark in test_loader:
            enc_x    = enc_x.float().to(device)
            dec_y_f  = dec_y.float()
            enc_mark = enc_mark.float().to(device)
            dec_mark = dec_mark.float().to(device)

            n_features = dec_y_f.shape[-1]

            # Decoder input: [label_len history ALL features | pred_len ZEROS ALL features]
            pred_pad = torch.zeros([dec_y_f.shape[0], cfg.PRED_LEN, n_features]).float()
            dec_inp  = torch.cat([dec_y_f[:, :cfg.LABEL_LEN, :], pred_pad], dim=1).float().to(device)

            out = model(enc_x, enc_mark, dec_inp, dec_mark)   # (B, pred_len, 1)
            gt  = dec_y_f[:, -cfg.PRED_LEN:, target_idx:target_idx+1]

            preds.append(inverse(out.squeeze(-1).cpu().numpy(), scaler, target_idx))
            trues.append(inverse(gt.squeeze(-1).cpu().numpy(), scaler, target_idx))

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return preds, trues


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Rebuild test set (same pipeline as training — no STL, no Winsorization) ──
    train_df, val_df, test_df = load_and_split(cfg.DATA_PATH, cfg.TRAIN_RATIO, cfg.VAL_RATIO)

    # Drop correlated columns
    if hasattr(cfg, 'DROP_COLS') and cfg.DROP_COLS:
        train_df = train_df.drop(columns=cfg.DROP_COLS, errors='ignore')
        val_df   = val_df.drop(columns=cfg.DROP_COLS, errors='ignore')
        test_df  = test_df.drop(columns=cfg.DROP_COLS, errors='ignore')

    # Time features
    train_df = add_time_features(train_df)
    val_df   = add_time_features(val_df)
    test_df  = add_time_features(test_df)

    # Scale
    scaler       = fit_scaler(train_df)
    test_scaled  = scale(test_df, scaler)
    col_names    = list(train_df.columns)
    target_idx   = col_names.index(cfg.TARGET_COL)
    test_index   = test_df.index

    print(f"  Columns ({len(col_names)}): {col_names}")
    print(f"  Target '{cfg.TARGET_COL}' at index {target_idx}")
    print(f"  Test shape: {test_scaled.shape}")

    results = {}
    s2s_preds = s2s_trues = inf_preds = inf_trues = None

    # ── Seq2Seq ───────────────────────────────────────────────────────────────
    if os.path.exists(cfg.S2S_CKPT):
        print("\nEvaluating Seq2SeqLSTM...")
        s2s_preds, s2s_trues = evaluate_seq2seq(test_scaled, scaler, target_idx, device)
        results["Seq2Seq"] = metric(s2s_preds, s2s_trues)
    else:
        print(f"Seq2Seq checkpoint not found: {cfg.S2S_CKPT}")

    # ── Informer ──────────────────────────────────────────────────────────────
    if os.path.exists(cfg.INF_CKPT):
        print("\nEvaluating Informer...")
        inf_preds, inf_trues = evaluate_informer(test_scaled, test_index, scaler, target_idx, device)
        results["Informer"] = metric(inf_preds, inf_trues)
    else:
        print(f"Informer checkpoint not found: {cfg.INF_CKPT}")

    # ── Print comparison table ─────────────────────────────────────────────────
    if results:
        print(f"\n{'='*55}")
        print(f"{'Model':<12} {'MAE':>8} {'RMSE':>8} {'MAPE':>8} {'MSE':>10}")
        print("-" * 55)
        for name, m in results.items():
            print(f"{name:<12} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} {m['MAPE']:>7.2f}% {m['MSE']:>10.4f}")
        print("=" * 55)

    # ── Plot first 3 × pred_len steps ─────────────────────────────────────────
    if results:
        os.makedirs(cfg.RESULT_DIR, exist_ok=True)
        n_show = cfg.PRED_LEN * 3

        fig, axes = plt.subplots(len(results), 1, figsize=(14, 4 * len(results)))
        if len(results) == 1:
            axes = [axes]

        for ax, (name, _) in zip(axes, results.items()):
            if name == "Seq2Seq" and s2s_preds is not None:
                preds_show = s2s_preds[:n_show, 0] if s2s_preds.ndim > 1 else s2s_preds[:n_show]
                trues_show = s2s_trues[:n_show, 0] if s2s_trues.ndim > 1 else s2s_trues[:n_show]
            elif name == "Informer" and inf_preds is not None:
                preds_show = inf_preds[:n_show, 0] if inf_preds.ndim > 1 else inf_preds[:n_show]
                trues_show = inf_trues[:n_show, 0] if inf_trues.ndim > 1 else inf_trues[:n_show]
            else:
                continue

            ax.plot(trues_show, label="Ground Truth", linewidth=1.2, color="#2196F3")
            ax.plot(preds_show, label=f"{name} Prediction", linewidth=1.2, linestyle="--", color="#FF5722")
            ax.set_title(f"{name} — first {n_show} prediction steps (original scale)")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("OT (°C)")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(cfg.RESULT_DIR, "comparison_plot.png")
        plt.savefig(save_path, dpi=150)
        plt.show()
        print(f"\nPlot saved to: {save_path}")


if __name__ == "__main__":
    main()
