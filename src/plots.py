"""
Plot helpers – generate matplotlib figures and return as base64 PNG strings
so they can be embedded directly in HTML templates.
"""

import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


PALETTE = {
    "seq2seq":   "#4C9BE8",
    "tcn":       "#F4845F",
    "attention": "#5CB85C",
    "true":      "#FFFFFF", # Changed to white for better visibility
    "grid":      "#ECF0F1",
    "bg":        "#1A1A2E",
    "panel":     "#16213E",
    "text":      "#E0E0E0",
}


def _apply_dark_style(fig, axes):
    """Apply dark theme to figure."""
    fig.patch.set_facecolor(PALETTE["bg"])
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(PALETTE["panel"])
        ax.tick_params(colors=PALETTE["text"])
        ax.xaxis.label.set_color(PALETTE["text"])
        ax.yaxis.label.set_color(PALETTE["text"])
        ax.title.set_color(PALETTE["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.grid(True, alpha=0.2, color="#555")


# ─────────────────────────────────────────────────────────────────────────────
# Learning curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_learning_curves(results: dict) -> str:
    """
    results = {
      "seq2seq": {"train": [...], "val": [...], "best_epoch": int},
      "tcn":     {...},
      "attention": {...},
    }
    Returns base64 PNG.
    """
    labels  = {"seq2seq": "LSTM Seq2Seq", "tcn": "TCN_v2"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _apply_dark_style(fig, axes)

    for ax, (key, res) in zip(axes, results.items()):
        ep = range(1, len(res["train"]) + 1)
        ax.plot(ep, res["train"], color=PALETTE[key], linewidth=2.2,
                label="Train MSE", alpha=0.9)
        ax.plot(ep, res["val"],   color="#E74C3C",   linewidth=2.2,
                linestyle="--", label="Val MSE", alpha=0.9)
        if "best_epoch" in res:
            ax.axvline(res["best_epoch"], color="#F1C40F", linestyle=":",
                       linewidth=1.6, label=f"Best ep={res['best_epoch']}")
        ax.set_title(labels[key], fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE (scaled)")
        ax.legend(fontsize=9, framealpha=0.3,
                  labelcolor=PALETTE["text"],
                  facecolor=PALETTE["panel"])

    fig.suptitle("Learning Curves – all models", fontsize=15,
                 fontweight="bold", color=PALETTE["text"], y=1.02)
    plt.tight_layout()
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction visualisation (first N samples)
# ─────────────────────────────────────────────────────────────────────────────

def plot_predictions(preds: dict, trues: dict, n_show: int = 400) -> str:
    """
    preds / trues = {"seq2seq": np.ndarray, "tcn": ..., "attention": ...}
    Shape of each array: (N, pred_len)  – first step only plotted per sample.
    """
    labels = {"seq2seq": "LSTM Seq2Seq", "tcn": "TCN_v2"}
    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
    _apply_dark_style(fig, axes)

    for ax, key in zip(axes, ["seq2seq", "tcn"]):
        p  = preds[key][:n_show, 0]
        t  = trues[key][:n_show, 0]
        ax.plot(t, color=PALETTE["true"],    linewidth=1.8, label="Ground Truth",  alpha=1.0)
        ax.plot(p, color=PALETTE[key],       linewidth=1.5, label=f"{labels[key]} Pred", alpha=0.85)
        mse = float(np.mean((p - t) ** 2))
        ax.set_title(f"{labels[key]} | MSE={mse:.4f} °C²",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("OT (°C)")
        ax.legend(fontsize=9, framealpha=0.3,
                  labelcolor=PALETTE["text"], facecolor=PALETTE["panel"])

    axes[-1].set_xlabel("Sample index")
    plt.tight_layout()
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Bar chart comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_bar_comparison(metrics: dict) -> str:
    """
    metrics = {"seq2seq": {MSE, RMSE, MAE, sMAPE%}, "tcn": ..., "attention": ...}
    """
    metric_keys  = ["MSE", "RMSE", "MAE", "sMAPE%"]
    model_keys   = ["seq2seq", "tcn"]
    labels       = ["LSTM Seq2Seq", "TCN_v2"]
    colors       = [PALETTE[k] for k in model_keys]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    _apply_dark_style(fig, axes)

    for ax, mkey in zip(axes, metric_keys):
        vals = [metrics[mk][mkey] for mk in model_keys]
        bars = ax.bar(labels, vals, color=colors, edgecolor="#333", linewidth=0.8)
        ax.set_title(mkey, fontsize=13, fontweight="bold")
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002 * max(vals),
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=9, color=PALETTE["text"])

    fig.suptitle("Test-Set Metric Comparison (°C units)", fontsize=14,
                 fontweight="bold", color=PALETTE["text"], y=1.02)
    plt.tight_layout()
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Interactive single-window prediction chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_multi_prediction(preds: dict, true: np.ndarray,
                           model_labels: dict) -> str:
    steps = np.arange(1, len(true) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_dark_style(fig, [ax])
    ax.plot(steps, true, color=PALETTE["true"], linewidth=2.5, marker="o",
            markersize=4, label="Ground Truth")
    
    for key, pred in preds.items():
        # use specific markers for each model to distinguish them
        marker = "s" if key == "seq2seq" else "^"
        ax.plot(steps, pred, color=PALETTE[key], linewidth=2.2, marker=marker,
                markersize=5, linestyle="--", label=f"{model_labels[key]} Pred")

    ax.set_xlabel("Forecast Step (15-min intervals)")
    ax.set_ylabel("OT (°C)")
    ax.set_title("All Models – 24-step Forecast", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.4,
              labelcolor=PALETTE["text"], facecolor=PALETTE["panel"])
    plt.tight_layout()
    return _fig_to_b64(fig)


def plot_actual_prediction(preds: dict, true: np.ndarray, dates, model_labels: dict) -> str:
    """
    Vẽ biểu đồ 3 đường so sánh: Thực tế (Ground Truth), LSTM Seq2Seq và TCN_v2.
    Trục X sẽ là mốc thời gian thực tế định dạng %H:%M.
    """
    fig, ax = plt.subplots(figsize=(10, 5.2))
    _apply_dark_style(fig, [ax])
    
    # Chuyển đổi dates thành định dạng chuỗi HH:MM
    x_labels = [d.strftime("%H:%M") for d in dates]
    steps = np.arange(len(true))
    
    ax.plot(steps, true, color=PALETTE["true"], linewidth=2.8, marker="o",
            markersize=5, label="Thực tế (Ground Truth)")
    
    for key, pred in preds.items():
        marker = "s" if key == "seq2seq" else "^"
        ax.plot(steps, pred, color=PALETTE[key], linewidth=2.2, marker=marker,
                markersize=6, linestyle="--", label=f"Dự báo {model_labels[key]}")

    # Đặt nhãn trục X là mốc thời gian
    ax.set_xticks(steps)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)

    ax.set_xlabel("Thời gian thực tế (Khoảng cách 15 phút)", fontsize=10, fontweight="bold", labelpad=8)
    ax.set_ylabel("OT (°C)", fontsize=10, fontweight="bold")
    
    start_time_str = dates[0].strftime("%d/%m/%Y %H:%M")
    ax.set_title(f"So sánh thực tế & dự báo 24 bước - Bắt đầu từ {start_time_str}", 
                 fontsize=12, fontweight="bold", pad=12)
                 
    ax.legend(fontsize=10, framealpha=0.4,
              labelcolor=PALETTE["text"], facecolor=PALETTE["panel"])
    plt.tight_layout()
    return _fig_to_b64(fig)

