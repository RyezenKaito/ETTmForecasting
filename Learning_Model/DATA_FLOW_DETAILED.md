# 🔄 DATA FLOW & COMPONENT BREAKDOWN - Seq2Seq LSTM

## 1. ENCODER: BiLSTM - Data Flow

```
INPUT SEQUENCE
================
(B=64, T=336, F=8)

[Timestep 1]        [Timestep 2]        [Timestep 3]  ...  [Timestep 336]
    x₁                  x₂                  x₃                  x₃₃₆
  (64,8)              (64,8)              (64,8)              (64,8)
    ↓                   ↓                   ↓                   ↓


FORWARD LSTM (→)
===============
x₁ → h₁_f (256D) → x₂ → h₂_f (256D) → x₃ → h₃_f (256D) → ... → h₃₃₆_f (256D)
                                                                     ↓
                                                        Final: (B, 512D)


BACKWARD LSTM (←)
===============
x₃₃₆ → h₃₃₆_b (256D) → ... → x₃ → h₃_b (256D) → x₂ → h₂_b (256D) → x₁ → h₁_b (256D)


CONCATENATE
===========
Per timestep i:
  h[i]_combined = cat[h[i]_forward, h[i]_backward]
                = (256D) + (256D)
                = 512D

ENC_OUT = [h₁_combined, h₂_combined, h₃_combined, ..., h₃₃₆_combined]
        = (B=64, T=336, 512)


FINAL STATES (for Decoder init)
==============================
h_combined = cat[h_forward_final, h_backward_final]
           = cat[(64, 256), (64, 256)]
           = (64, 512)

c_combined = cat[c_forward_final, c_backward_final]
           = (64, 512)

----- OUTPUT -----
enc_out: (64, 336, 512) ← for Attention
h: (64, 512)            ← initialize Decoder hidden
c: (64, 512)            ← initialize Decoder cell
```

### Encoder Purpose
- **Captures full sequence context** (forward + backward)
- **All outputs available** for Attention mechanism
- **Final state** gives summary of entire input

### Parameters: ~640K
```
2 LSTM layers, bidirectional
Layer 1: input=8 → hidden=256
  params = 4 × 256 × (8 + 256 + 1) × 2 directions ≈ 271K

Layer 2: input=256 → hidden=256
  params = 4 × 256 × (256 + 256 + 1) × 2 directions ≈ 525K

Total: ~640K
```

---

## 2. ATTENTION: Scaled Dot-Product - Data Flow

```
INPUT
=====
Decoder hidden state at step t:
  h[t] = (B=64, 512D)

Encoder outputs (same for all t):
  enc_out = (B=64, T=336, 512D)


STEP 1: CREATE QUERY
===================
Q = h[t].unsqueeze(1)
  = (B, 512) → (B, 1, 512)

Interpretation: "What is decoder looking for?"


STEP 2: COMPUTE SCORES
=====================
scores = Q @ enc_out.transpose(1, 2) × (1/√512)
       = (B, 1, 512) @ (B, 512, 336)
       = (B, 1, 336)

per_timestep_score = [score₁, score₂, score₃, ..., score₃₃₆]

Example scores (before scale):
  [2.3, 0.5, 5.2, 1.1, -0.3, 0.8, ...]
                ↑
         Encoder timestep 3 is relevant!

After scale (1/√512 ≈ 0.044):
  [0.10, 0.02, 0.23, 0.05, -0.01, 0.04, ...]

Why scale?
- Without: large values → softmax near 0/1 → no gradient
- With: moderate values → softmax smooth → good gradient


STEP 3: SOFTMAX → WEIGHTS
=========================
weights = softmax(scores, dim=-1)
        = (B, 1, 336)

Example:
  Input:  [0.10, 0.02, 0.23, 0.05, -0.01, 0.04, ...]
  Softmax: [0.08, 0.02, 0.34, 0.06, 0.01, 0.03, ...]
                           ↑
                    Focus 34% on step 3

All weights sum to 1.0: Σweights = 1.0


STEP 4: APPLY TO VALUES
=======================
context = weights @ enc_out
        = (B, 1, 336) @ (B, 336, 512)
        = (B, 1, 512)
        → squeeze(1) → (B, 512)

context[i] = Σ(weight[i][t] × enc_out[i][t])
           = 0.08×enc_out[:,0] + 0.02×enc_out[:,1] + 0.34×enc_out[:,2] + ...

Output shape: (B, 512) ← ready to combine with decoder input


----- OUTPUT -----
context: (B, 512) ← "What encoder says is relevant"
```

### Attention Purpose
- **Dynamic focus:** learns what input timesteps matter for each prediction
- **Interpretable:** weights show model's "attention"
- **Flexible:** different queries → different weights (not fixed)

### Parameters: ~5K
```
No learnable parameters!
Only matrix computations:
  - Matrix multiply: Q @ K^T
  - Scale factor: 1/√512 (constant)
  - Softmax: non-parametric
  - Weighted sum: @ V

~5K = overhead, mostly edge effects
```

---

## 3. DECODER: Autoregressive Loop - Data Flow

```
INITIALIZE (after Encoder)
=========================
h, c = from encoder final states  (B, 512)
prev_out = x[:, -1, 0]           (B, 1)  — last input value
outputs = []


LOOP: FOR t = 0 TO 23
====================

┌─ TIMESTEP t=0
│
├─ INPUT PREPARATION
│  ┌──────────────────────────────
│  │ prev_out = x[:, -1, 0] = (B, 1)
│  │   ↓
│  │ future_cov[t] = future_cov[:, 0, :] = (B, 4)
│  │   ├─ time_sin: sin(2π × t / 96)
│  │   ├─ time_cos: cos(2π × t / 96)
│  │   ├─ day_sin: sin(2π × dow / 7)
│  │   └─ day_cos: cos(2π × dow / 7)
│  │   ↓
│  │ context = attention(h, enc_out) = (B, 512)
│  │   ↓
│  │ dec_in = cat[prev_out, future_cov, context]
│  │        = (B, 1) + (B, 4) + (B, 512)
│  │        = (B, 517)
│  │
│  │ Shape breakdown:
│  │ [prev_value: 1D | time_feats: 4D | encoder_context: 512D]
│  │ [    0.5°C    | sin/cos/sin/cos  |   weighted history  ]
│  └──────────────────────────────
│  ↓
│
├─ LSTM FORWARD
│  ┌──────────────────────────────
│  │ (h[t], c[t]) = LSTM_layer_1(dec_in, h[t-1], c[t-1])
│  │                LSTM_layer_2(lstm1_out, h[t-1], c[t-1])
│  │
│  │ Input: (B, 1, 517)
│  │ Output: (B, 1, 512)
│  │
│  │ LSTM learns:
│  │ - Input gate: how much to update h
│  │ - Forget gate: what to forget
│  │ - Cell gate: what new info to add
│  │ - Output gate: what h to output
│  │
│  │ h[t] = (B, 512) ← accumulates information
│  │ c[t] = (B, 512) ← internal memory
│  └──────────────────────────────
│  ↓
│
├─ OUTPUT HEAD
│  ┌──────────────────────────────
│  │ hidden = h[t].squeeze(1) = (B, 512)
│  │   ↓
│  │ x = LayerNorm(hidden) = (B, 512)
│  │   ↓
│  │ x = Linear(512→128)(x) = (B, 128)
│  │   ↓
│  │ x = GELU(x) = (B, 128)  [non-linear]
│  │   ↓
│  │ x = Dropout(0.2)(x) = (B, 128)  [training only]
│  │   ↓
│  │ pred[t] = Linear(128→1)(x) = (B, 1)
│  │
│  │ Output: temperature prediction for hour t
│  └──────────────────────────────
│  ↓
│
├─ TEACHER FORCING DECISION
│  ┌──────────────────────────────
│  │ tf_ratio = 0.6 × 0.98^epoch
│  │          = [0.6 epoch 1 → 0.1 epoch 100]
│  │
│  │ if training and random() < tf_ratio:
│  │     prev_out = y[:, t]  ← USE GROUND TRUTH (60%→10%)
│  │     [Learn from correct examples]
│  │ else:
│  │     prev_out = pred[t]  ← USE PREDICTION (40%→90%)
│  │     [Learn from own mistakes]
│  │
│  │ Why decay?
│  │ - Early training: learn from examples
│  │ - Late training: learn self-sufficiency
│  └──────────────────────────────
│  ↓
│
└─ SAVE OUTPUT
   outputs.append(pred[t])
   
   After loop iterations...
   t = 0: pred[0] = (B, 1)
   t = 1: pred[1] = (B, 1)
   ...
   t = 23: pred[23] = (B, 1)


CONCATENATE ALL OUTPUTS
=======================
predictions = cat([pred[0], pred[1], ..., pred[23]], dim=1)
            = (B, 24)

Each batch element: 24-hour forecast
```

### Decoder Purpose
- **Sequential generation:** predict step-by-step
- **Autoregressive:** each prediction depends on previous
- **Teacher Forcing:** helps training converge faster
- **Flexible length:** can predict any number of steps (not just 24)

### Parameters: ~4.2M

```
LSTM Layer 1: input=517 → hidden=512
  params = 4 × 512 × (517 + 512 + 1)
         = 4 × 512 × 1030
         = 2,109,440

LSTM Layer 2: input=512 → hidden=512
  params = 4 × 512 × (512 + 512 + 1)
         = 4 × 512 × 1025
         = 2,099,200

Total: 2,109,440 + 2,099,200 = 4,208,640 ≈ 4.2M

WHY SO LARGE?
- Input 517D (vs Encoder 8D) → 65× larger input space
- Full LSTM params proportional to input size
- 2 layers × large input = expensive
```

---

## 4. OUTPUT HEAD: Dense Transformation - Data Flow

```
INPUT from Decoder LSTM
=======================
h[t] from 2-layer LSTM = (B, 512)


LAYERNORM
=========
x = LayerNorm(h[t])
  = (h[t] - mean(h[t])) / √(var(h[t]) + ε) × γ + β

where γ (weight) and β (bias) are learned.

Purpose: normalize distribution, stabilize training
Output: (B, 512)


LINEAR 512→128
==============
x = Linear(512, 128)(x)
  = x @ W.T + b

where W is (512, 128) weight matrix, b is (128,) bias

This is a learnable projection/transformation.
Could also think: "extract top 128 features from 512"

Output: (B, 128)

Parameters: 512×128 + 128 = 65,664


GELU ACTIVATION
===============
x = GELU(x)
  = x × Φ(x)

where Φ is the cumulative Gaussian distribution.

Purpose: non-linearity, learn complex relationships
Smoother than ReLU, better gradients

Output: (B, 128)

Parameters: 0 (non-parametric)


DROPOUT
=======
x = Dropout(0.2)(x)

Training: randomly zero 20% of values
Evaluation: no dropout (all values pass)

Purpose: prevent co-adaptation, regularization

Output: (B, 128)

Parameters: 0 (no learnable params)


LINEAR 128→1
============
pred = Linear(128, 1)(x)
     = x @ W.T + b

where W is (128, 1), b is (1,)

Final transformation: (B, 128) → (B, 1) prediction

Output: (B, 1)

Parameters: 128×1 + 1 = 129


----- FINAL OUTPUT -----
pred[t] = (B, 1) ← temperature prediction for hour t
```

### Output Head Purpose
- **Feature reduction:** 512D → 128D (bottleneck)
- **Non-linearity:** learn complex dependencies
- **Regularization:** dropout prevents overfitting
- **Final mapping:** dense representation → scalar prediction

### Parameters: ~67K

```
LayerNorm:    1,024  (512 weight + 512 bias)
Linear 512→128: 65,664  (512×128 + 128)
Linear 128→1:   129     (128×1 + 1)

Total: 66,817 ≈ 67K
```

---

## 5. CONCATENATE & OUTPUT - Data Flow

```
DECODER OUTPUTS (24 timesteps)
==============================
pred[0]: (B, 1) = (64, 1)  ← hour 1 prediction
pred[1]: (B, 1) = (64, 1)  ← hour 2 prediction
pred[2]: (B, 1) = (64, 1)  ← hour 3 prediction
...
pred[23]: (B, 1) = (64, 1)  ← hour 24 prediction


CONCATENATE
===========
output = torch.cat([pred[0], pred[1], ..., pred[23]], dim=1)
       = cat[(64,1), (64,1), ..., (64,1)]
       = (64, 24)

Shape breakdown:
  - Dimension 0 (64): batch size — different sequences
  - Dimension 1 (24): prediction timesteps — next 24 hours


INVERSE SCALING
===============
For evaluation (convert back to °C):

pred_celsius = pred_scaled × scaler.scale_[0] + scaler.mean_[0]

where:
  pred_scaled: model output (normalized)
  scaler.scale_[0]: std of OT in training data
  scaler.mean_[0]: mean of OT in training data

Example:
  pred_scaled = 0.5 (normalized)
  mean = 18.5°C
  std = 5.0°C
  → pred_celsius = 0.5 × 5.0 + 18.5 = 21°C


----- FINAL OUTPUT -----
output: (B, 24) = (64, 24)

Each row: 24-hour forecast
  [20.1°C, 20.5°C, 21.2°C, ..., 19.8°C]
```

### Purpose
- **Stack predictions:** all 24 steps into single tensor
- **Inverse scale:** convert model outputs back to original units
- **Ready for evaluation:** compare with ground truth

---

## 6. COMPLETE PIPELINE SUMMARY

```
┌──────────────────────────────────────────────────────────────┐
│                        COMPLETE FLOW                         │
└──────────────────────────────────────────────────────────────┘

TRAINING:
=========
Input (64, 336, 8)
  ↓ [640K params]
Encoder BiLSTM
  enc_out (64, 336, 512), h,c (64, 512)
  ↓ [5K params]
Loop × 24 steps
  ├─ Attention: (512) → (512)
  ├─ Decoder LSTM: (517) → (512) [4.2M params]
  ├─ Output Head: (512) → (1) [67K params]
  └─ Teacher Forcing: use y[t] or pred[t]
  ↓
Output (64, 24)
  ↓
Loss = MSE(pred, y) + 0.3×MSE(diff_pred, diff_y)
  ↓
Backward + Optimize
  ↓
Scheduler.step() — reduce learning rate


INFERENCE (Evaluation):
=======================
Input (64, 336, 8)
  ↓
Encoder BiLSTM
  enc_out (64, 336, 512), h,c (64, 512)
  ↓
Loop × 24 steps
  ├─ Attention: (512) → (512)
  ├─ Decoder LSTM: (517) → (512)
  ├─ Output Head: (512) → (1)
  └─ Teacher Forcing OFF: always use pred[t]
  ↓
Output (64, 24)
  ↓
Inverse Scale: (64, 24) scaled → (64, 24) °C
  ↓
Evaluate: MSE, RMSE, MAE, sMAPE vs ground truth


KEY DIFFERENCES (Train vs Inference):
====================================
Training:
  - tf_ratio = 0.6 → 0.1 (scheduled)
  - Use y[t] 60%→10% of time
  - Input noise for robustness
  - Dropout active

Inference:
  - tf_ratio = 0.0 (always use predictions)
  - Pure autoregressive
  - No noise
  - Dropout disabled
```

---

## 7. EXAMPLE: Actual Data Through Pipeline

```
CONCRETE EXAMPLE: Predict next 24 hours
======================================

Input Sequence (1 batch element):
  Days: [Day 1, Day 2, Day 3, Day 4, Day 5, Day 6, Day 7]
  Hours: 96 hours/day × 7 = 672 values, last 336 selected
  
  x shape: (1, 336, 8)
  x values (pseudo):
    x[0,0,:] = [20.5, 19.2, 0.31, 0.44, 0.11, 0.99, -0.10, 0.99]  ← time 0
    x[0,1,:] = [20.4, 19.2, 0.32, 0.44, 0.12, 0.99, -0.09, 0.99]  ← time 1
    ...
    x[0,335,:] = [21.2, 20.1, 0.39, 0.42, 0.88, 0.48, 0.47, 0.88]  ← time 335


ENCODER:
Forward LSTM sees: [x₀, x₁, ..., x₃₃₅]
  → learns 0→336 pattern: "temperature rising over week"

Backward LSTM sees: [x₃₃₅, x₃₃₄, ..., x₀]
  → learns context from future: "fits into weekly cycle"

enc_out[0, :, :] = (336, 512)
  Each timestep has 512D representation capturing context
  
h[0, :] = (512,)  — summary of entire week
c[0, :] = (512,)  — internal memory


DECODER LOOP (t=0 to 23):

t=0 (predicting hour 24):
  prev_out = x[0, 335, 0] = 21.2  ← last observed temp
  
  future_cov[0, 0, :] = [sin(4π/96), cos(4π/96), sin(2π/7), cos(2π/7)]
                      = [0.196, 0.981, 0.445, 0.896]
                      ← features for hour 24
  
  context = attention(h[0], enc_out[0])
          = weighted sum of 336 encoder outputs
          = [focus: recent temps 40%, same-hour-last-week 35%, ...]
          = (512,)
  
  dec_in = cat[[21.2], [0.196, 0.981, 0.445, 0.896], context]
         = (517,)
  
  LSTM processes dec_in, returns h[1] = (512,)
  
  Dense head transforms h[1] → pred[0]
  
  If training & random() < 0.6:
      prev_out = y[0]  ← ground truth temp for hour 24
  else:
      prev_out = pred[0]  ← model's prediction


t=1 (predicting hour 25):
  prev_out = y[0] or pred[0]  ← from t=0
  future_cov[0, 1, :] = features for hour 25
  context = weighted_sum(enc_out)  ← different weights
  dec_in = cat[prev_out, future_cov[1], context]
  
  LSTM processes, Dense head → pred[1]


...continue for t=2 to t=23...


Final Output:
  preds[0] = [pred[0], pred[1], ..., pred[23]]
           = [21.5, 21.8, 22.1, 22.3, 22.2, 21.9, 21.5, ...]
           = (1, 24)
           
Inverse scale:
  preds_celsius = preds * 5.0 + 18.5  (example: std=5, mean=18.5)
                = [21.5°C, 21.8°C, 22.1°C, ..., 21.5°C]


Ground Truth:
  y[0] = [21.4, 21.9, 22.0, 22.4, 22.1, 21.8, 21.6, ...]
  
Metrics:
  MSE = mean((preds - y)²)
      = mean([0.01, 0.01, 0.01, ...])
      = ~0.02
  
  RMSE = √0.02 = 0.141°C
  MAE = mean(|preds - y|)
      = 0.11°C
```

---

## 8. DIMENSIONS CHEATSHEET

| Stage | Shape | Meaning |
|-------|-------|---------|
| Input | (B, T, F) = (64, 336, 8) | Batch of 64, 336 timesteps, 8 features |
| Encoder out | (B, T, 2H) = (64, 336, 512) | All hidden states from BiLSTM |
| Encoder state | (B, 2H) = (64, 512) | Final hidden + cell state |
| Attention query | (B, 1, 2H) = (64, 1, 512) | Decoder state at step t |
| Attention context | (B, 2H) = (64, 512) | Weighted sum of encoder |
| Decoder input | (B, 1+4+2H) = (64, 517) | prev_out + cov + context |
| Decoder LSTM out | (B, 2H) = (64, 512) | LSTM hidden state |
| Dense head out | (B, 1) = (64, 1) | Prediction for step t |
| Final output | (B, pred_len) = (64, 24) | 24-hour forecast |

---

## 9. PARAMETER SIZE INTUITION

Why is Decoder so much larger (4.2M vs 640K)?

```
SIZE = f(input_dimension, hidden_size, num_layers)

SIZE ∝ input × hidden × num_layers

Encoder:
  input = 8
  hidden = 256
  layers = 1-2
  → 8 × 256 = 2,048 units of computation
  → ~640K params

Decoder:
  input = 517
  hidden = 512
  layers = 2
  → 517 × 512 = 264,704 units!
  → 517/8 = 65× larger input
  → 512/256 = 2× larger hidden
  → 2 layers
  → Total: 65 × 2 × 2 ≈ 260× more params
  
Actual: 4.2M / 0.64M ≈ 6.5× ✓
  (less than 260× because encoder has 2 directions doubling its size)
```

