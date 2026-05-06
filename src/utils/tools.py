"""
src/utils/tools.py
Training utilities: EarlyStopping, learning-rate scheduler helper, plotting.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when val loss stops improving for `patience` epochs."""

    def __init__(self, patience: int = 5, verbose: bool = True, delta: float = 0.0):
        self.patience   = patience
        self.verbose    = verbose
        self.delta      = delta
        self.counter    = 0
        self.best_score = None
        self.early_stop = False
        self.best_val   = float("inf")

    def __call__(self, val_loss: float, model: torch.nn.Module, path: str):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save(val_loss, model, path)
            self.counter = 0

    def _save(self, val_loss: float, model: torch.nn.Module, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)
        if self.verbose:
            print(f"  Val loss improved to {val_loss:.4f}  -> checkpoint saved.")
        self.best_val = val_loss


# ─────────────────────────────────────────────────────────────────────────────
# Plot training history
# ─────────────────────────────────────────────────────────────────────────────

def plot_history(history, title: str = "Training Loss", save_path: str = None):
    """
    history: list of (train_loss, val_loss) tuples or dict with those keys.
    """
    if isinstance(history, list):
        train_losses = [h[0] for h in history]
        val_losses   = [h[1] for h in history]
    else:
        train_losses = history["train"]
        val_losses   = history["val"]

    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses,   label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (original scale)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()
