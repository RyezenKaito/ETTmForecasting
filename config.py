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

# STL & Winsorization
STL_PERIOD  = 96               # 96 × 15 min = 1 day

# Columns to drop (high correlation)
DROP_COLS = ["MUFL", "MULL"]

# Feature count after preprocessing (with DROP_COLS applied):
# Raw cols: HUFL, HULL, LUFL, LULL, OT               → 5
# + trend, seasonal, residual (from STL on OT)        → +3
# + time_sin, time_cos                                → +2
# Total                                               → 10
N_FEATURES  = 10

# ─── Seq2Seq Hyperparameters ───────────────────────────────────────────────────
S2S_HIDDEN_SIZE = 64           # Reduced from 128 to combat overfitting
S2S_NUM_LAYERS  = 2
S2S_DROPOUT     = 0.5          # Increased from 0.3 for stronger regularization
S2S_LR          = 5e-4         # Slower LR for more stable convergence
S2S_WEIGHT_DECAY= 1e-3         # Increased from 1e-4 for stronger L2 regularization
S2S_EPOCHS      = 80
S2S_BATCH_SIZE  = 64
S2S_PATIENCE    = 5            # More patience to allow convergence
S2S_CKPT        = os.path.join(CKPT_DIR, "seq2seq_best.pth")

# ─── Informer Hyperparameters ─────────────────────────────────────────────────
# Decoder only sees OT, sin, cos
DEC_IN_DIM   = 3

INF_ENC_IN   = N_FEATURES      # 9
INF_DEC_IN   = DEC_IN_DIM      # 3
INF_C_OUT    = 1               # Output features (MS mode: predict only OT)
INF_D_MODEL  = 64              # Reduced from 128 to combat overfitting
INF_N_HEADS  = 2               # Reduced from 4
INF_E_LAYERS = 1               # Reduced from 2
INF_D_LAYERS = 1
INF_D_FF     = 128             # Reduced from 512
INF_FACTOR   = 3               # ProbSparse attention factor
INF_DROPOUT  = 0.5             # Increased from 0.3 — stronger regularization
INF_ATTN     = "prob"          # "prob" = ProbSparse, "full" = vanilla
INF_EMBED    = "timeF"         # Time-feature embedding
INF_FREQ     = "t"             # minutely (15min)
INF_ACTIVATION = "gelu"
INF_OUTPUT_ATTENTION = False
INF_DISTIL   = True
INF_MIX      = True
INF_PADDING  = 0
INF_LR       = 5e-5            # Halved from 1e-4 — slower, more stable
INF_WEIGHT_DECAY = 1e-3        # Increased from 1e-4 — stronger L2 regularization
INF_EPOCHS   = 30              # Increased from 10 — more room to converge
INF_BATCH_SIZE = 64            # Increased from 32 — smoother gradients
INF_PATIENCE = 5               # Increased from 3 — less aggressive early stop
INF_CKPT     = os.path.join(CKPT_DIR, "informer_best.pth")
