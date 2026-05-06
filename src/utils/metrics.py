"""
src/utils/metrics.py
Regression evaluation metrics used for comparing Seq2Seq vs Informer.
"""

import numpy as np


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean((pred - true) ** 2))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(mse(pred, true)))


def mape(pred: np.ndarray, true: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs((pred - true) / (np.abs(true) + eps))) * 100)


def metric(pred: np.ndarray, true: np.ndarray):
    """Return dict of all metrics."""
    return {
        "MAE":  mae(pred, true),
        "MSE":  mse(pred, true),
        "RMSE": rmse(pred, true),
        "MAPE": mape(pred, true),
    }
