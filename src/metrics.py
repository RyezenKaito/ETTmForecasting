"""
Metric helpers shared across scripts and the Flask app.
"""

import numpy as np


def inverse_target(x, scaler, idx):
    """Inverse-scale only the target column."""
    return x * scaler.scale_[idx] + scaler.mean_[idx]


def calc_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    """Return MSE, RMSE, MAE, sMAPE% (all in original units if inputs are)."""
    mse  = float(np.mean((y_pred - y_true) ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(y_pred - y_true)))
    denom = (np.abs(y_pred) + np.abs(y_true)) / 2.0
    denom = np.maximum(denom, 1.0)
    smape = float(np.mean(np.abs(y_pred - y_true) / denom) * 100)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "sMAPE%": smape}
