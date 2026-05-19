"""
ETTm1 Multi-step Forecasting Demo – Flask Application
Run:  python app.py
"""

import os
import json
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify

# ── local modules ─────────────────────────────────────────────────────────────
from src.preprocess import build_pipeline, SEQ_LEN, LABEL_LEN, PRED_LEN, N_COV
from src.models import MODEL_CONFIGS
from src.metrics import inverse_target, calc_metrics
from src.inference import (TimeSeriesDataset, load_model,
                            evaluate_model, predict_sample)
from src.plots import (plot_learning_curves, plot_predictions,
                       plot_bar_comparison, plot_multi_prediction)
from torch.utils.data import DataLoader

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data", "ETTm1.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
ARTS_DIR    = os.path.join(BASE_DIR, "artifacts")
RESULTS_DIR = os.path.join(ARTS_DIR, "results")

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── global state (loaded once at startup) ─────────────────────────────────────
_PIPELINE   = None   # (train_sc, val_sc, test_sc, scaler, ti, nf, train_df)
_MODELS     = {}     # {key: model}
_TEST_DS    = None
_TEST_PREDS = {}     # {key: np.ndarray (N, pred_len)}
_TEST_TRUES = {}
_METRICS    = {}     # {key: {MSE, RMSE, MAE, sMAPE%}}
_LC_DATA    = {}     # learning-curve stub


def _init():
    global _PIPELINE, _MODELS, _TEST_DS, _TEST_PREDS, _TEST_TRUES, _METRICS, _LC_DATA

    if _PIPELINE is not None:
        return  # already initialised

    print("[init] Building data pipeline…")
    (train_sc, val_sc, test_sc,
     scaler, ti, nf, train_df) = build_pipeline(DATA_PATH)
    _PIPELINE = (train_sc, val_sc, test_sc, scaler, ti, nf, train_df)

    print(f"[init] Columns: {list(train_df.columns)}")
    print(f"[init] target_index={ti}, n_features={nf}")
    print(f"[init] Scaler mean[OT]={scaler.mean_[ti]:.4f}, scale[OT]={scaler.scale_[ti]:.4f}")

    _TEST_DS = TimeSeriesDataset(test_sc, SEQ_LEN, LABEL_LEN, PRED_LEN)
    test_loader = DataLoader(_TEST_DS, batch_size=64, shuffle=False)

    # Try to load pre-computed results first
    results_path = os.path.join(RESULTS_DIR, "metrics_all_models.json")
    preds_available = os.path.exists(results_path)

    if preds_available:
        print("[init] Loading pre-computed metrics from artifacts/results/…")
        with open(results_path) as f:
            _METRICS = json.load(f)
        for key in MODEL_CONFIGS:
            p_path = os.path.join(RESULTS_DIR, f"predictions_{key}.npy")
            t_path = os.path.join(RESULTS_DIR, "trues.npy")
            if os.path.exists(p_path) and os.path.exists(t_path):
                _TEST_PREDS[key] = np.load(p_path)
                _TEST_TRUES[key] = np.load(t_path)

    # Always load models (needed for /api/predict and /weights)
    for key in MODEL_CONFIGS:
        if key not in _MODELS:
            try:
                m = load_model(key, MODELS_DIR, nf, PRED_LEN, ti, device)
                _MODELS[key] = m
            except Exception as e:
                print(f"  WARNING: could not load {key}: {e}")

    # If no pre-computed results, evaluate now
    if not preds_available:
        print("[init] No pre-computed results – running evaluation…")
        for key in MODEL_CONFIGS:
            if key in _MODELS:
                try:
                    preds, trues, met = evaluate_model(
                        _MODELS[key], key, test_loader, ti, PRED_LEN, scaler, device)
                    _TEST_PREDS[key] = preds
                    _TEST_TRUES[key] = trues
                    _METRICS[key]    = met
                    print(f"    {key} MSE={met['MSE']:.4f}")
                except Exception as e:
                    print(f"  WARNING: eval failed for {key}: {e}")

    # Learning-curve data (best-epoch info from notebook logs)
    _LC_DATA = {
        "seq2seq":   {"train": [], "val": [], "best_epoch": 2,  "best_val": 0.3154},
        "tcn":       {"train": [], "val": [], "best_epoch": 5,  "best_val": 0.1622},
    }

    print("[init] Done.")


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    _init()
    return render_template("index.html",
                           metrics=_METRICS,
                           model_configs=MODEL_CONFIGS,
                           device=str(device))


@app.route("/pipeline")
def pipeline():
    _init()
    _, _, _, scaler, ti, nf, train_df = _PIPELINE
    return render_template("pipeline.html",
                           columns=list(train_df.columns),
                           n_features=nf,
                           target_index=ti,
                           scaler_mean=round(scaler.mean_[ti], 4),
                           scaler_scale=round(scaler.scale_[ti], 4))


@app.route("/models")
def models_page():
    _init()
    _, _, _, _, _, nf, _ = _PIPELINE
    return render_template("models.html",
                           n_features=nf,
                           model_configs=MODEL_CONFIGS)


@app.route("/results")
def results():
    _init()
    bar_chart = plot_bar_comparison(_METRICS) if len(_METRICS) == 2 else None
    pred_chart = None
    if len(_TEST_PREDS) == 2:
        pred_chart = plot_predictions(_TEST_PREDS, _TEST_TRUES)
    return render_template("results.html",
                           metrics=_METRICS,
                           bar_chart=bar_chart,
                           pred_chart=pred_chart,
                           lc_data=_LC_DATA)


@app.route("/visualize")
def visualize():
    _init()
    _, _, test_sc, _, _, _, _ = _PIPELINE
    n_samples = max(0, len(test_sc) - SEQ_LEN - PRED_LEN)
    return render_template("visualize.html",
                           n_samples=n_samples,
                           model_keys=list(MODEL_CONFIGS.keys()),
                           model_labels={k: v["label"]
                                         for k, v in MODEL_CONFIGS.items()})


@app.route("/weights")
def weights():
    _init()
    weight_report = {}
    for key, model in _MODELS.items():
        layers = []
        total  = 0
        for name, param in model.named_parameters():
            n = param.numel()
            total += n
            layers.append({
                "name":  name,
                "shape": list(param.shape),
                "n":     n,
                "norm":  round(float(param.data.norm(2)), 4),
                "mean":  round(float(param.data.mean()), 6),
                "std":   round(float(param.data.std()),  6),
            })
        weight_report[key] = {
            "layers": layers,
            "total":  total,
            "label":  MODEL_CONFIGS[key]["label"],
        }
    return render_template("weights.html", report=weight_report)


# ── API ───────────────────────────────────────────────────────────────────────

@app.route("/api/metrics")
def api_metrics():
    _init()
    return jsonify(_METRICS)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    _init()
    body      = request.get_json(force=True)
    idx       = int(body.get("index", 0))

    _, _, test_sc, scaler, ti, _, _ = _PIPELINE
    
    all_preds = {}
    all_metrics = {}
    model_labels = {}
    
    true_vals = None
    
    for key, model in _MODELS.items():
        pred, true = predict_sample(model, key, test_sc, idx,
                                    PRED_LEN, ti, scaler, device)
        all_preds[key] = pred
        all_metrics[key] = calc_metrics(pred, true)
        model_labels[key] = MODEL_CONFIGS[key]["label"]
        if true_vals is None:
            true_vals = true

    if not all_preds:
        return jsonify({"error": "No models available"}), 400

    chart = plot_multi_prediction(all_preds, true_vals, model_labels)

    # Convert arrays to list for JSON
    preds_json = {k: v.tolist() for k, v in all_preds.items()}

    return jsonify({
        "preds":   preds_json,
        "true":    true_vals.tolist(),
        "metrics": all_metrics,
        "chart":   chart,
    })


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5500, debug=True)
