"""
scripts/run_evaluation.py
Run offline (once) to generate artifacts/results/*.npy and metrics_all_models.json
Usage: python scripts/run_evaluation.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.preprocess import build_pipeline, SEQ_LEN, LABEL_LEN, PRED_LEN
from src.models import MODEL_CONFIGS
from src.inference import TimeSeriesDataset, load_model, evaluate_model

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data", "ETTm1.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "artifacts", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("Building pipeline...")
(train_sc, val_sc, test_sc,
 scaler, ti, nf, sp, train_df) = build_pipeline(DATA_PATH)

test_ds = TimeSeriesDataset(test_sc, SEQ_LEN, LABEL_LEN, PRED_LEN)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

all_metrics = {}
all_trues   = None

for key in MODEL_CONFIGS:
    print(f"\nEvaluating {key}...")
    try:
        model, mdim = load_model(key, MODELS_DIR, nf, PRED_LEN, ti, device)
        print(f"  Loaded with input_dim={mdim}")
        preds, trues, metrics = evaluate_model(
            model, key, test_loader, ti, PRED_LEN, scaler, device, model_dim=mdim)
        print(f"  MSE={metrics['MSE']:.4f}  RMSE={metrics['RMSE']:.4f}  "
              f"MAE={metrics['MAE']:.4f}  sMAPE={metrics['sMAPE%']:.2f}%")
        np.save(os.path.join(RESULTS_DIR, f"predictions_{key}.npy"), preds)
        all_metrics[key] = metrics
        if all_trues is None:
            all_trues = trues
    except Exception as e:
        import traceback
        print(f"  ERROR: {e}")
        traceback.print_exc()

if all_trues is not None:
    np.save(os.path.join(RESULTS_DIR, "trues.npy"), all_trues)

with open(os.path.join(RESULTS_DIR, "metrics_all_models.json"), "w") as f:
    json.dump(all_metrics, f, indent=2)

print("\nDone! Artifacts saved to artifacts/results/")
print(json.dumps(all_metrics, indent=2))
