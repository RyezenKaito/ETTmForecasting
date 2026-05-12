"""
config.py — Centralized hyperparameters and settings for ETT Forecasting project.
All scripts import from here to keep configuration in a single place.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(ROOT_DIR, "data", "ETTm1.csv")
CKPT_DIR    = os.path.join(ROOT_DIR, "checkpoints")
RESULT_DIR  = os.path.join(ROOT_DIR, "results")

# ─── Dataset ──────────────────────────────────────────────────────────────────
TARGET_COL  = "OT"
FREQ        = "15min"          # ETTm1 is sampled every 15 minutes

# Sequence lengths
SEQ_LEN     = 96               # Encoder input length (lookback window: 1 day)
LABEL_LEN   = 48               # = SEQ_LEN // 2  — Informer decoder "start token"
PRED_LEN    = 24               # Forecast horizon

# Train / Val / Test split ratios (60 / 20 / 20)
TRAIN_RATIO = 0.6
VAL_RATIO   = 0.2
# TEST_RATIO is implicitly 1 - TRAIN_RATIO - VAL_RATIO = 0.2

# Columns to drop (high correlation)
DROP_COLS = ["MUFL", "MULL"]

# Feature count after preprocessing (with DROP_COLS applied):
# Raw cols: HUFL, HULL, LUFL, LULL, OT               → 5
# + time_sin, time_cos                                → +2
# Total                                               → 7
N_FEATURES  = 7

# Decoder input dim for Seq2Seq (autoregressive: [OT, sin, cos])
S2S_DEC_IN_DIM = 3

# ─── Seq2Seq Hyperparameters ───────────────────────────────────────────────────
S2S_HIDDEN_SIZE = 128          # Increased from 64 for better capacity
S2S_NUM_LAYERS  = 2
S2S_DROPOUT     = 0.3          # Reduced from 0.5 — less aggressive
S2S_LR          = 3e-4         # Slightly lower for stability
S2S_WEIGHT_DECAY= 1e-4         # Light L2 regularization
S2S_EPOCHS      = 100
S2S_BATCH_SIZE  = 64
S2S_PATIENCE    = 7            # More patience to allow convergence
S2S_CKPT        = os.path.join(CKPT_DIR, "seq2seq_best.pth")

# ─── Informer Hyperparameters ─────────────────────────────────────────────────
# IMPORTANT: dec_in = enc_in = N_FEATURES (matches the original paper)
# Decoder receives ALL features; the future portion is zero-padded (no leakage).
INF_ENC_IN   = N_FEATURES      # 7
INF_DEC_IN   = N_FEATURES      # 7 (same as enc_in — per original paper)
INF_C_OUT    = 1               # Output features (MS mode: predict only OT)
INF_D_MODEL  = 256             # Increased from 64 (paper uses 512, 256 is balanced)
INF_N_HEADS  = 4               # Increased from 2 (minimum for effective attention)
INF_E_LAYERS = 2               # Increased from 1 (need ≥2 for distilling to work)
INF_D_LAYERS = 1
INF_D_FF     = 1024            # = 4 × d_model (standard Transformer ratio)
INF_FACTOR   = 5               # ProbSparse attention factor (paper default)
INF_DROPOUT  = 0.05            # Reduced from 0.5 — paper uses 0.05
INF_ATTN     = "prob"          # "prob" = ProbSparse, "full" = vanilla
INF_EMBED    = "timeF"         # Time-feature embedding
INF_FREQ     = "t"             # minutely (15min)
INF_ACTIVATION = "gelu"
INF_OUTPUT_ATTENTION = False
INF_DISTIL   = True
INF_MIX      = True
INF_PADDING  = 0
INF_LR       = 1e-4            # Paper default
INF_WEIGHT_DECAY = 1e-5        # Light L2 regularization (paper uses 0)
INF_EPOCHS   = 50              # More room to converge
INF_BATCH_SIZE = 32            # Smaller batch for noisier but more frequent updates
INF_PATIENCE = 7               # Less aggressive early stopping
INF_CKPT     = os.path.join(CKPT_DIR, "informer_best.pth")
