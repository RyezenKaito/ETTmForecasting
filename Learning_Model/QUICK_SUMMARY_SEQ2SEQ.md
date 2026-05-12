# 🚀 QUICK SUMMARY: Seq2SeqLSTM Architecture

## 📊 Tổng Quan Kiến Trúc

```
INPUT (B, 336, 8)
    ↓
┌─────────────────────────────────────┐
│  ENCODER: BiLSTM (Bidirectional)    │
│  - Forward LSTM: 256D                │
│  - Backward LSTM: 256D               │
│  - Output: (B, 336, 512)             │
│  - State: h,c (B, 512)               │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  DECODER LOOP (24 bước)             │
│  ┌───────────────────────────────┐  │
│  │ t=0: prev_out → Attention →   │  │
│  │       Context → LSTM → pred[0]│  │
│  │       (TF: use y[0] or pred)  │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │ t=1: prev_out → Attention →   │  │
│  │       Context → LSTM → pred[1]│  │
│  │       (TF: use y[1] or pred)  │  │
│  └───────────────────────────────┘  │
│  ... (22 bước nữa) ...              │
└─────────────────────────────────────┘
    ↓
OUTPUT (B, 24)
```

---

## 🧠 4 Thành Phần Chính

### 1️⃣ **S2SEncoder: BiLSTM Encoder**

```python
class S2SEncoder(nn.Module):
    self.lstm = LSTM(input_dim, hidden, bidirectional=True)
    
forward(x):  # (B, 336, 8)
    enc_out, (h, c) = self.lstm(x)
    h = cat[forward_h, backward_h]  # (B, 512)
    c = cat[forward_c, backward_c]  # (B, 512)
    return enc_out (B, 336, 512), h, c
```

**Ý nghĩa:**
- BiLSTM = 2 LSTM chạy cùng lúc (thuận & ngược)
- Nắm bắt context từ cả hai hướng
- `enc_out`: tất cả hidden states (dùng cho Attention)
- `h, c`: cuối state (dùng khởi tạo Decoder)

---

### 2️⃣ **S2SAttention: Scaled Dot-Product Attention**

```python
class S2SAttention(nn.Module):
    scale = 1 / sqrt(hidden*2)  # 1/sqrt(512) ≈ 0.044
    
forward(h_last, enc_out):  # h_last: (B, 512), enc_out: (B, 336, 512)
    q = h_last.unsqueeze(1)  # (B, 1, 512)
    scores = q @ enc_out^T * scale  # (B, 1, 336)
    weights = softmax(scores)  # sum to 1.0
    context = weights @ enc_out  # (B, 1, 512) → squeeze → (B, 512)
    return context
```

**Công thức:** `Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V`

**Ý nghĩa:**
- Query: "Decoder đang tìm cái gì?"
- Scores: "Mỗi phần input có liên quan bao nhiêu?"
- Weights: Xác suất chú ý vào mỗi input timestep
- Context: Tổng có trọng số của input states

**Ví dụ:**
```
Encoder: [x₁, x₂, x₃, ..., x₃₃₆]  (336 timesteps)
Weights: [0.01, 0.34, 0.15, ..., 0.01]  (tổng = 1.0)
Context: 0.01×x₁ + 0.34×x₂ + 0.15×x₃ + ... 
→ Decoder tập trung 34% vào x₂!
```

---

### 3️⃣ **S2SDecoder: LSTM Decoder with Attention**

```python
class S2SDecoder(nn.Module):
    self.attn = S2SAttention(hidden)
    self.lstm = LSTM(dec_in + hidden*2, hidden*2)  # (5+512, 512)
    self.fc = Linear(512 → 128 → 1)
    
forward(dec_in, h, c, enc_out):  # dec_in: (B, 1, 5)
    ctx = self.attn(h[-1], enc_out)  # (B, 512)
    inp = cat[dec_in, ctx]  # (B, 1, 517)
    out, (h, c) = self.lstm(inp, (h, c))  # (B, 1, 512)
    pred = self.fc(out)  # (B, 1)
    return pred, h, c
```

**Ý nghĩa:**
- Nhận: previous output + time features + encoder states
- Tính attention context
- LSTM cập nhật trạng thái
- Dense head output 1 giá trị

---

### 4️⃣ **Seq2SeqLSTM: Main Model (Autoregressive Loop)**

```python
class Seq2SeqLSTM(nn.Module):
    def forward(self, x, y=None, future_cov=None, tf_ratio=0.5):
        # Encoding
        enc_out, h, c = self.encoder(x)
        
        # Decoding loop
        prev_out = x[:, -1, 0]  # Last value của target
        outputs = []
        
        for t in range(24):  # 24 steps
            # Concat input + time features
            cov_t = future_cov[:, t, :]  # (B, 4)
            dec_in = cat[prev_out, cov_t]  # (B, 5)
            
            # Decode một bước
            pred, h, c = self.decoder(dec_in, h, c, enc_out)
            outputs.append(pred)
            
            # Teacher Forcing
            if training and y is not None and random() < tf_ratio:
                prev_out = y[:, t]  # Use ground truth
            else:
                prev_out = pred  # Use prediction
        
        return cat(outputs)  # (B, 24)
```

---

## 🎯 Teacher Forcing

**Training (tf_ratio = 0.6 → 0.1):**
```
Step 0: input=y[0] (60% TF) → output=pred[0] → next=y[1] hoặc pred[0]
Step 1: input=... → output=pred[1] → ...
...
```
- **Tại sao?** Học từ correct examples, convergence nhanh
- **Decay schedule:** `tf = max(0.6 * 0.98^epoch, 0.1)`
  - Epoch 1: 60% ground truth
  - Epoch 50: 35% ground truth
  - Epoch 100: 10% ground truth
  - **Lý do:** Từ từ phụ thuộc vào predictions → exposure bias mitigation

**Inference (tf_ratio = 0.0):**
```
Step 0: input=x[-1] → output=pred[0] → next=pred[0]
Step 1: input=pred[0] (có thể sai!) → output=pred[1] → ...
→ Error accumulation (autoregressive)
```

---

## 📈 Hyperparameters

| Parameter | Giá trị | Ý nghĩa |
|-----------|--------|---------|
| `input_dim` | 8 | OT + 3 STL + 4 time features |
| `hidden_size` | 256 | LSTM units (→ 512 sau concat) |
| `n_layers` | 2 | LSTM layers |
| `dropout` | 0.2 | Regularization |
| `dec_in_dim` | 5 | prev_out (1D) + time_features (4D) |
| `pred_len` | 24 | Predict 24 timesteps ahead |
| `batch_size` | 64 | Batch size |
| `learning_rate` | 1e-4 | Initial LR |
| `optimizer` | AdamW | with L2=1e-3 |
| `scheduler` | CosineAnnealingLR | LR: 1e-4 → 1e-6 |

**Model Size:**
```
Total parameters: ~1.26M
├─ Encoder: ~640k
├─ Attention: ~5k
└─ Decoder: ~611k
```

---

## 🔄 Data Flow Example

```
Input: (64, 336, 8)
├─ 64 batches
├─ 336 timesteps (7 days)
└─ 8 features [OT, trend, seasonal, residual, time_sin, time_cos, day_sin, day_cos]

↓ Encoder

enc_out: (64, 336, 512) ← tất cả encoder states
h: (64, 512) ← hidden state cuối
c: (64, 512) ← cell state cuối

↓ Decoder Loop (24 bước)

t=0:
  ├─ prev_out: (64, 1) = x[:, -1, 0]
  ├─ future_cov[:, 0]: (64, 4) = [time_sin, time_cos, day_sin, day_cos]
  ├─ Attention(h, enc_out): (64, 512)
  ├─ LSTM(cat[prev, cov, ctx]): (64, 512)
  └─ Dense: (64, 1)

t=1...23:
  (tương tự, h,c được update)

↓ Concatenate

Output: (64, 24)
```

---

## 💡 Tại Sao Kiến Trúc Này Tốt?

✅ **BiLSTM:** Full context (forward + backward)
✅ **Attention:** Model tự học tập trung vào đâu (interpretable!)
✅ **Teacher Forcing:** Convergence nhanh, training ổn định
✅ **Autoregressive:** Flexible, có thể dự báo bất kỳ độ dài nào
✅ **Time Features:** Nắm bắt seasonality (hour, day of week)

---

## ⚠️ Thách Thức

❌ **Sequential:** Phải dự báo từng bước → chậm (24 steps × LSTM)
❌ **Error Accumulation:** Inference (autoregressive) có thể drift
❌ **Vanishing Gradient:** RNN khó train, nhưng LSTM + attention giúp

---

## 🎓 So Sánh với TCN_v2 (Cái khác trong notebook)

| Aspect | Seq2Seq LSTM | TCN_v2 |
|--------|-------------|--------|
| **Architecture** | RNN | CNN |
| **Receptive Field** | Tất cả (via recurrence) | Hạn chế (via dilation) |
| **Attention** | ✓ Có | ✗ Không |
| **Speed** | Chậm (sequential) | Nhanh (parallel) |
| **Interpretability** | Cao (attention weights) | Thấp |
| **Suited For** | Capturing long-term deps | Efficient inference |

---

## 🔗 Loss Function

```python
loss = MSE_loss(pred, y_true) + TREND_LAMBDA * trend_loss(pred, y_true)

where:
  MSE_loss = L2 loss on values
  trend_loss = MSE(diff[pred], diff[y_true])
            = MSE on gradients (slopes)
  TREND_LAMBDA = 0.3
  
→ 70% dự báo values chính xác
→ 30% dự báo trends (rate of change) chính xác
```

**Tại sao?** Không chỉ dự báo giá trị, mà cả xu hướng thay đổi

---

## 📚 Kết Luận

**Seq2SeqLSTM with Attention** là một mô hình mạnh mẽ cho time series forecasting bởi vì:

1. **Encoder nắm toàn bộ context** → BiLSTM 2 hướng
2. **Decoder biết tập trung đâu** → Attention mechanism
3. **Training ổn định** → Teacher Forcing with decay
4. **Dự báo flexible** → Autoregressive (có thể predict any length)
5. **Interpretable** → Xem attention weights để hiểu model suy nghĩ gì

👍 **Dùng khi:** Cần capture long-term dependencies, cần hiểu model
👎 **Không dùng:** Khi cần real-time inference (quá chậm), cần massive parallelization

---

**📖 Để hiểu chi tiết hơn**, xem file `GIAI_THICH_LSTM_SEQ2SEQ_ATTENTION.md`!

