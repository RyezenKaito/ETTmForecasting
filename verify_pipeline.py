import os, sys, numpy as np, torch
from torch.utils.data import DataLoader

from src.preprocess import build_pipeline, SEQ_LEN, LABEL_LEN, PRED_LEN
from src.inference import TimeSeriesDataset, load_model, evaluate_model, predict_sample
from src.metrics import calc_metrics

DATA_PATH  = "data/ETTm1.csv"
MODELS_DIR = "models"
device = torch.device("cpu")

print("=== Building pipeline ===")
train_sc, val_sc, test_sc, scaler, ti, nf, train_df = build_pipeline(DATA_PATH)
print("Columns:", list(train_df.columns))
print("target_index=%d, n_features=%d" % (ti, nf))
print("Scaler mean[OT]=%.4f, scale[OT]=%.4f" % (scaler.mean_[ti], scaler.scale_[ti]))
print("test_sc shape:", test_sc.shape)

test_ds = TimeSeriesDataset(test_sc, SEQ_LEN, LABEL_LEN, PRED_LEN)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
print("Test samples:", len(test_ds))

for key in ["seq2seq", "tcn"]:
    print("\n=== Loading %s ===" % key)
    model = load_model(key, MODELS_DIR, nf, PRED_LEN, ti, device)
    
    print("=== Evaluating %s ===" % key)
    preds, trues, met = evaluate_model(model, key, test_loader, ti, PRED_LEN, scaler, device)
    print("  MSE=%.4f, RMSE=%.4f, MAE=%.4f" % (met["MSE"], met["RMSE"], met["MAE"]))
    
    # Test sample 4575
    pred, true = predict_sample(model, key, test_sc, 4575, PRED_LEN, ti, scaler, device)
    sample_mse = np.mean((pred - true)**2)
    print("  Sample #4575 MSE=%.4f" % sample_mse)
    print("  Pred[:5]:", pred[:5].round(2))
    print("  True[:5]:", true[:5].round(2))

print("\nDONE!")
