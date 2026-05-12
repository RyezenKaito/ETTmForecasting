# ⚡ QUICK REFERENCE: Seq2Seq LSTM Architecture

## 🏗️ ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────┐
│                    SEQ2SEQ LSTM PIPELINE                │
└─────────────────────────────────────────────────────────┘

Input (B, 336, 8)
    ↓
    ├─ Batch: 64 sequences
    ├─ History: 336 timesteps (7 days)
    └─ Features: 8 (OT + STL + time)

ENCODER
    ├─ Component: BiLSTM (forward + backward)
    ├─ Layers: 2
    ├─ Hidden: 256 per direction = 512 concatenated
    ├─ Parameters: ~640K
    └─ Output:
        ├─ enc_out: (64, 336, 512) ← all states
        ├─ h: (64, 512) ← final hidden
        └─ c: (64, 512) ← final cell

ATTENTION (× 24 loop iterations)
    ├─ Component: Scaled Dot-Product
    ├─ Formula: softmax(Q @ K^T / √d) @ V
    ├─ Parameters: ~5K (mostly no-op)
    └─ Output: context (64, 512)

DECODER (Autoregressive loop, 24 steps)
    ├─ LSTM Layer:
    │  ├─ Input: prev_output(1) + time_features(4) + context(512) = 517D
    │  ├─ Hidden: 512
    │  ├─ Layers: 2
    │  └─ Parameters: ~4.2M
    │
    ├─ Dense Head:
    │  ├─ LayerNorm(512)
    │  ├─ Linear(512→128) + GELU + Dropout(0.2)
    │  ├─ Linear(128→1)
    │  └─ Parameters: ~67K
    │
    ├─ Teacher Forcing:
    │  ├─ Training: use ground truth 60%→10% (decay)
    │  └─ Inference: use predictions 100%
    │
    └─ Output: pred[0..23] (each B, 1)

Concatenate: [pred[0], pred[1], ..., pred[23]]
    ↓
Output (B, 24) ← 24-hour forecast

TOTAL PARAMETERS: ~4.9M (640K + 5K + 4.2M + 67K)
```

---

## 🔢 PARAMETER BREAKDOWN

```
┌──────────────────┬──────────┬────────────┐
│ Component        │ Params   │ Percentage │
├──────────────────┼──────────┼────────────┤
│ Encoder BiLSTM   │ ~640K    │ 13%        │
│ Attention        │ ~5K      │ 0.1%       │
│ Decoder LSTM     │ ~4.2M    │ 86%        │
│ Output Head      │ ~67K     │ 1.4%       │
├──────────────────┼──────────┼────────────┤
│ TOTAL            │ ~4.9M    │ 100%       │
└──────────────────┴──────────┴────────────┘

Memory: ~20 MB (weights) + ~80 MB (activations) = ~100 MB
        (with optimizer states: ~200-250 MB total)
```

---

## 🎯 EACH COMPONENT'S JOB

### 1️⃣ ENCODER (BiLSTM) - 640K params

**What it does:**
```
Take: 336 hours of history (all 8 features)
Return: 512D summary of each hour + final states

Process:
  Forward:  →→→→ left to right (capture forward trend)
  Backward: ←←←← right to left (capture backward context)
  Concat: (256 + 256) = 512D per timestep
```

**Formula:**
```
params = 4 × hidden × (input + hidden + 1) × num_layers × num_directions
       = 4 × 256 × (8 + 256 + 1) × 2 × 2 = ~640K
```

**Why important:**
- Full context from past (forward) + future context for past (backward)
- All 336 states available for Attention
- Transfer information to Decoder via h, c

---

### 2️⃣ ATTENTION - 5K params

**What it does:**
```
Query: "What are we looking for?" (decoder state)
Keys/Values: "Where are relevant things?" (encoder outputs)
Output: Weighted selection of encoder info

Example:
  Query = "predicting hour 20"
  Weights = [0.02, 0.34, 0.05, 0.01, ..., 0.01]
  Context = 0.02×enc[0] + 0.34×enc[1] + ... ← weighted sum
```

**Formula:**
```
scores = Q @ K^T / √512
       = (64, 1, 512) @ (64, 512, 336) / 0.044
       = (64, 1, 336)

weights = softmax(scores)  ← all sum to 1.0
context = weights @ V = (64, 512)
```

**Why important:**
- Model learns what encoder parts matter
- Interpretable: see attention weights
- Different queries → different attention (flexible)

---

### 3️⃣ DECODER LSTM - 4.2M params

**What it does:**
```
Loop 24 times:
  1. Get attention context
  2. Concatenate: prev_output + time_features + context
  3. Pass through LSTM
  4. Dense head → prediction
  5. Teacher Forcing: next input = y[t] or pred[t]
```

**Input composition:**
```
dec_in = cat[
    prev_output(1D),          ← what we predicted/know
    time_features(4D),        ← hour, day_of_week
    context(512D)             ← what encoder says is relevant
] = 517D
```

**Formula:**
```
Layer 1: params = 4 × 512 × (517 + 512 + 1) = 2.1M
Layer 2: params = 4 × 512 × (512 + 512 + 1) = 2.1M
Total: 4.2M

Why large?
  - Input 517D (vs Encoder 8D) = 65× more
  - Hidden 512 (vs Encoder 256) = 2× more
  - 2 layers × large input = expensive
```

**Teacher Forcing:**
```
Training (tf_ratio = 0.6 → 0.1):
  60% of time: use y[t] (ground truth)
  40% of time: use pred[t] (model prediction)
  
  Epoch 1: 60% ground truth, 40% model
  Epoch 50: 35% ground truth, 65% model
  Epoch 100: 10% ground truth, 90% model
  
  Benefit: Learn from examples → gradually self-sufficient

Inference (tf_ratio = 0.0):
  Always use pred[t] (pure autoregressive)
  Potential error accumulation
```

---

### 4️⃣ OUTPUT HEAD - 67K params

**What it does:**
```
512D hidden state → 1D prediction

LayerNorm(512)
  ↓
Linear(512 → 128)    ← compress
  ↓
GELU()               ← non-linearity
  ↓
Dropout(0.2)         ← regularization
  ↓
Linear(128 → 1)      ← final prediction
```

**Formula:**
```
LayerNorm: 1,024 params (512 weight + 512 bias)
Linear(512→128): 65,664 params (512×128 + 128)
Linear(128→1): 129 params (128×1 + 1)
Total: 66,817 ≈ 67K
```

**Why important:**
- Reduce dimensionality (512D → 128D bottleneck)
- Non-linear transformation (GELU)
- Regularization (Dropout prevents overfitting)
- Final mapping to scalar

---

## 📊 DATA FLOW

```
┌─────────────────────────────────────────────┐
│ Training                                    │
├─────────────────────────────────────────────┤
│                                             │
│ Input (64, 336, 8)                         │
│   ↓ [Encoder: ~640K]                       │
│ enc_out (64, 336, 512), h,c (64, 512)      │
│   ↓ [Loop 24 times]                        │
│   ├─ Attention: (64, 512) → (64, 512)      │
│   ├─ Decoder LSTM: (64, 517) → (64, 512)   │
│   ├─ Dense Head: (64, 512) → (64, 1)       │
│   └─ TF: use y[t] (60%→10%) or pred[t]     │
│   ↓                                         │
│ Output (64, 24)                            │
│   ↓ [Loss]                                 │
│ loss = MSE(pred, y) + 0.3×MSE(∇pred, ∇y)   │
│   ↓ [Backward + Optimize]                  │
│ Update weights                             │
│   ↓ [Scheduler]                            │
│ LR: 1e-4 → 1e-6 (cosine annealing)         │
│                                             │
├─────────────────────────────────────────────┤
│ Inference                                   │
├─────────────────────────────────────────────┤
│                                             │
│ Input (64, 336, 8)                         │
│   ↓                                         │
│ [Same pipeline but]                        │
│   └─ TF: use pred[t] always (100%)         │
│   ↓                                         │
│ Output (64, 24)                            │
│   ↓ [Inverse Scale]                        │
│ Output_celsius (64, 24)                    │
│   ↓ [Evaluate vs ground truth]             │
│ MSE, RMSE, MAE, sMAPE                      │
│                                             │
└─────────────────────────────────────────────┘
```

---

## 🧮 DIMENSION TRACKING

```
After Encoder:
  enc_out: (B, T, 512) = (64, 336, 512)
  h: (B, 512) = (64, 512)
  c: (B, 512) = (64, 512)

Decoder Loop (for each t):
  prev_out: (B, 1) = (64, 1)
  time_cov: (B, 4) = (64, 4)
  context: (B, 512) = (64, 512)
  dec_in: (B, 1+4+512) = (64, 517)
  
  LSTM in: (B, 1, 517)
  LSTM out: (B, 1, 512)
  
  Dense: (B, 512) → (B, 1)
  pred[t]: (B, 1)

Final:
  cat all 24 preds
  Output: (B, 24) = (64, 24)
```

---

## 📐 FORMULAS

```
LSTM Parameters:
  P = 4 × H × (I + H + 1)
  where H=hidden, I=input
  
Linear Parameters:
  P = I × O + O
  where I=input, O=output
  
Attention:
  Q = query (decoder state)
  K = keys (encoder states)
  V = values (encoder states)
  A = softmax(Q @ K^T / √d) @ V
  
Loss:
  L = MSE(pred, true) + λ × MSE(∇pred, ∇true)
  where λ = 0.3 (trend weight)
  
Teacher Forcing Ratio:
  tf = max(0.6 × 0.98^epoch, 0.1)
  epoch 1: 60%
  epoch 50: 35%
  epoch 100: 10%
```

---

## ⏱️ COMPUTATIONAL COST

```
Model Size:
  Weights: 4.9M params × 4 bytes = ~20 MB
  Activations (b=64): ~60-80 MB
  Total: ~100 MB
  + Optimizer states: ~200 MB
  
Inference Time (batch=64):
  Encoder: ~1 ms
  Attention × 24: ~9 ms
  Decoder × 24: ~12 ms
  Total: ~22 ms per batch
  
Training Time (per epoch):
  ~500 batches × 1ms = ~500 ms
  100 epochs ≈ 50 seconds
  
GPU Memory:
  ~200-250 MB (RTX 3060: 12GB → plenty)
```

---

## 🎓 KEY CONCEPTS

| Concept | What It Means |
|---------|---------------|
| **BiLSTM** | LSTM forward + backward = full context |
| **Attention** | Learn what to focus on (interpretable) |
| **Autoregressive** | Each prediction depends on previous |
| **Teacher Forcing** | Use ground truth in training, predictions at inference |
| **Scheduled Decay** | tf_ratio: 60% → 10% over epochs (gradual independence) |
| **Trend Loss** | Penalize wrong rate-of-change, not just wrong values |
| **Dropout** | Randomly disable 20% of neurons (prevents overfitting) |
| **LayerNorm** | Normalize distributions (stable training) |
| **GELU** | Smooth activation (better gradients than ReLU) |
| **Cosine Annealing** | LR: 1e-4 → 1e-6 (smooth decay schedule) |

---

## ⚙️ HYPERPARAMETER SETTINGS

```
Architecture:
  input_dim: 8
  hidden_size: 256
  n_layers: 2
  dropout: 0.2
  dec_in_dim: 5 (prev_out + 4 time features)
  pred_len: 24

Data:
  batch_size: 64
  seq_len: 336 (7 days)
  split: 60% train, 20% val, 20% test

Training:
  epochs: 100
  optimizer: AdamW
  lr: 1e-4
  weight_decay: 1e-3
  scheduler: CosineAnnealingLR (eta_min=1e-6)
  
Loss:
  mse_weight: 0.7
  trend_weight: 0.3
  grad_clip: 1.0
  
Early Stopping:
  patience: 5 epochs
  
Regularization:
  dropout: 0.2
  noise_std: 0.02 (input noise)
  
Teacher Forcing:
  init_ratio: 0.6
  min_ratio: 0.1
  decay: 0.98 (per epoch)
```

---

## 🔍 DEBUGGING CHECKLIST

```
If training doesn't converge:
  ☐ Check learning rate (1e-4 usually good)
  ☐ Check gradient clipping (1.0)
  ☐ Check for NaN in loss (numerical issues?)
  ☐ Reduce batch size (more frequent updates)
  ☐ Increase L2 weight decay (more regularization)

If overfitting:
  ☐ Increase dropout (0.2 → 0.3-0.5)
  ☐ Add more input noise (0.02 → 0.05)
  ☐ Increase trend_lambda (0.3 → 0.5)
  ☐ Larger weight_decay (1e-3 → 1e-2)

If underfitting:
  ☐ Increase hidden_size (256 → 512)
  ☐ Add more layers (2 → 3)
  ☐ Increase training epochs
  ☐ Decrease dropout rate

If slow inference:
  ☐ Use KV cache (cache encoder output)
  ☐ Batch process (64 at once, not 1)
  ☐ Quantize model (fp32 → int8)
  ☐ Reduce seq_len (336 → 168)
```

---

## 📚 SUMMARY

**Seq2Seq LSTM = Encoder-Decoder with Attention**

- **Encoder**: Understand history (BiLSTM, ~640K)
- **Attention**: Learn what matters (no params, interpretable)
- **Decoder**: Predict future step-by-step (large LSTM, ~4.2M)
- **Head**: Convert to scalar prediction (~67K)

**Total: ~4.9M parameters**
**Best for: Time series forecasting with long-term dependencies**

