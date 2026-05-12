# 🚀 BỘ 3 MODEL TỐI ƯU - HƯỚNG DẪN ΧIẾT TIẾT

## 📊 ĐỀ BÀI VẤN ĐỀ

**Tình trạng hiện tại (sau fix data leakage):**
- ❌ Predictions (đường đỏ) hoàn toàn sai so với actual (đường xanh)
- ❌ Model bị underfit hoặc architecture không phù hợp
- ❌ MSE rất lớn (0.75 → 33.15 → 16.82 tùy sample)

**Nguyên nhân:**
1. Sau khi fix leakage, model mất đi thông tin "cheat" từ future target
2. Các model cũ (BL-S2S, TCN_v2, LSTM-S2S) không đủ mạnh để học pattern mà không leakage
3. Hyperparameter tuning không tối ưu
4. Thiếu regularization và noise augmentation

---

## 🔧 3 MODEL ĐƯỢC CHỌN & LÝ DO

### 1️⃣ **TRANSFORMER** 
**Lợi ích:**
- ✅ Self-attention cho phép capture long-range dependencies
- ✅ Parallel processing (nhanh hơn RNN)
- ✅ Giải quyết vanishing gradient problem
- ✅ Tốt cho dữ liệu có pattern lặp lại

**Kiến trúc:**
```
Input → Linear Projection → Positional Encoding → 
Transformer Encoder (3 layers, 4 heads) → 
Transformer Decoder → Output Projection → Predictions
```

**Tại sao phù hợp:**
- ETTm1 có seasonal pattern rõ (period=96)
- Attention mechanism có thể học relationship giữa các timesteps

**Hyperparameters tối ưu:**
```
d_model=64 (dimension của attention)
nhead=4 (4 heads cho 64 dims = 16 dims per head - good balance)
num_layers=3 (3 encoder + 3 decoder layers)
dim_feedforward=256 (inner layer size)
dropout=0.2 (regularization)
lr=1e-3 (learning rate)
```

---

### 2️⃣ **GRU ENCODER-DECODER**
**Lợi ích:**
- ✅ GRU nhanh hơn LSTM (3 gates vs 4 gates)
- ✅ Better gradient flow (giải quyết vanishing gradient)
- ✅ Bidirectional encoder capture cả quá khứ và context
- ✅ Attention mechanism trong decoder
- ✅ Hoạt động tốt với sequence-to-sequence tasks

**Kiến trúc:**
```
Input → Bidirectional GRU Encoder → 
Context Vector (last hidden state) →
GRU Decoder (with Attention over encoder outputs) →
Output Projection → Predictions
```

**Tại sao phù hợp:**
- GRU sinh ra được để handle vanishing gradient tốt hơn
- Bidirectional encoder học cả forward và backward context
- Attention giúp focus vào relevant parts

**Hyperparameters tối ưu:**
```
hidden_dim=128 (hidden state size)
num_layers=2 (encoder + decoder layers)
dropout=0.2
lr=1e-3
Bidirectional attention (num_heads=4)
```

---

### 3️⃣ **IMPROVED TEMPORAL CONVOLUTIONAL NETWORK (TCN)**
**Lợi ích:**
- ✅ Receptive field lớn (dilation exponential: 2^0, 2^1, 2^2, ...)
- ✅ Parallel training (CNN trainable in parallel)
- ✅ Không bị vanishing gradient như RNN
- ✅ Residual connections cho deep networks
- ✅ Layer normalization + GELU activation

**Kiến trúc:**
```
Input → Dilated Conv Block 1 (dilation=1) →
Dilated Conv Block 2 (dilation=2) →
Dilated Conv Block 3 (dilation=4) →
Dilated Conv Block 4 (dilation=8) →
Dilated Conv Block 5 (dilation=16) →
Output Head → Predictions
```

**Receptive Field Computation:**
```
RF = 1 + 2 * (k - 1) * sum(2^i for i in range(n))
   = 1 + 2 * (7 - 1) * (1 + 2 + 4 + 8 + 16)
   = 1 + 2 * 6 * 31
   = 373 >> 336 (seq_len) ✓
```

**Tại sao phù hợp:**
- TCN có receptive field lớn hơn seq_len
- Parallel training nhanh hơn LSTM/GRU
- Residual blocks tránh degradation
- Cho dữ liệu time series có pattern dài

**Hyperparameters tối ưu:**
```
channels=[32, 64, 128, 128, 256] (exponential growth)
kernel_size=7 (receptive field lớn)
dropout=0.3 (stronger regularization)
lr=5e-4 (slightly lower for stability)
```

---

## 🛠️ CÁC CẢI TIẾN CHÍNH

### **1. Fix Data Leakage Hoàn Chỉnh**

#### ❌ Cách sai (trong notebook cũ):
```python
# Cell 44 - SAI: Dùng y_true_actual để tạo future features
df_future['OT'] = y_true_actual
df_future['trend'] = apply_trend(df_future)  # Dùng OT (leaked!)
df_future['seasonal'] = apply_seasonal(df_future)  # Dùng OT (leaked!)
```

#### ✅ Cách đúng (trong code mới):
```python
# Trong bộ 3 model này:
# - Seasonal pattern được fit ONLY trên train
# - Apply pattern đó cho val/test (không dùng future OT)
# - Trend được tính từ rolling mean của available data (quá khứ)
# - Không có information leakage từ future

for split_df in [val_df, test_df]:
    split_df['trend'] = apply_trend(split_df, window=period)  # OK: rolling trên split đó
    split_df['seasonal'] = apply_seasonal(split_df, seasonal_pattern)  # OK: pattern từ train
    split_df['residual'] = (split_df[target_col] - split_df['trend'] - split_df['seasonal']).values
```

---

### **2. Noise Augmentation**

**Vấn đề:** Model có thể overfit nếu không có regularization

**Giải pháp:** Thêm Gaussian noise vào input layer
```python
if use_noise and noise_std > 0:
    Xb = Xb + torch.randn_like(Xb) * noise_std  # noise_std=0.01
```

**Lợi ích:**
- Tăng robustness của model
- Tự động regularization
- Model học feature quan trọng thay vì noise

---

### **3. Better Optimization**

#### ✅ Cosine Annealing Learning Rate Schedule
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=1e-6
)
```

**Tại sao tốt hơn fixed LR?**
- LR bắt đầu từ 1e-3, dần giảm theo curve cosine
- Cho phép model hội tụ nhanh ở đầu, fine-tune ở cuối
- Tránh loss bị stuck ở local minimum

#### ✅ AdamW Optimizer
```python
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=lr, 
    weight_decay=1e-3  # L2 regularization
)
```

**Tại sao tốt hơn Adam?**
- AdamW decouples weight decay từ gradient-based update
- Tránh "AMSGrad" artifacts
- Better generalization

#### ✅ Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

**Lợi ích:**
- Tránh exploding gradients
- Ổn định training

---

### **4. Better Architecture Choices**

#### Positional Encoding (Transformer)
```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
```

**Tại sao cần?**
- Transformer không có notion của time order
- Positional encoding cung cấp relative position information

#### Layer Normalization vs Batch Normalization
```python
# Sử dụng LayerNorm thay vì BatchNorm
self.norm = nn.LayerNorm(d_model)
```

**Tại sao tốt hơn?**
- LayerNorm không phụ thuộc vào batch size
- Ổn định training với small batches
- Tốt cho time series

#### GELU Activation
```python
self.act = nn.GELU()  # Thay vì ReLU
```

**Tại sao tốt hơn ReLU?**
- GELU smooth hơn ReLU
- Không bị dead ReLU problem
- Generalize tốt hơn

---

### **5. Ensemble**

**Cách hoạt động:**
```python
ensemble_preds = (transformer_preds + gru_preds + tcn_preds) / 3
```

**Lợi ích:**
- Combine các model khác nhau
- Mỗi model capture khác nhau patterns
- Reduce variance
- Thường đạt SOTA results

---

## 📈 EXPECTED IMPROVEMENTS

### Comparison với notebook cũ:

| Aspect | Cũ | Mới | Cải tiến |
|--------|-----|-----|---------|
| **Leakage** | ❌ Có | ✅ Không | Critical fix |
| **Architecture** | LSTM + TCN | Transformer + GRU + TCN | Diversity |
| **LR Schedule** | Fixed | Cosine Annealing | Better convergence |
| **Regularization** | Dropout only | Dropout + Noise + Weight decay | Stronger |
| **Activation** | ReLU | GELU | Better gradients |
| **Normalization** | BatchNorm | LayerNorm | Stable |
| **Output Strategy** | Single model | Ensemble (3 models) | Lower variance |
| **Expected MSE** | ~2-33 | ~0.3-1.5 | 10-20x better |

---

## 🎯 HƯỚNG DẪN SỬ DỤNG

### **Step 1: Run code**
```bash
python optimized_models.py
```

### **Step 2: Check outputs**
- `learning_curves_optimized.png` - Training curves (should show convergence)
- `predictions_optimized.png` - Sample predictions (should match actual better)
- `metrics_comparison_optimized.png` - Metrics comparison

### **Step 3: Interpret results**

**Good signs:**
- ✅ Val loss decrease và level off
- ✅ Val/Train ratio < 1.2 (không overfit)
- ✅ Predictions (red) gần actual (blue)
- ✅ Ensemble MSE < individual models

**Bad signs:**
- ❌ Val loss keep increasing → overfitting
- ❌ Val/Train ratio >> 1 → need more regularization
- ❌ Predictions flat hoặc sai hoàn toàn → underfit

---

## 🔍 DEBUGGING TIPS

Nếu kết quả không tốt:

### 1️⃣ **Model still underfit (predictions flat)?**
```python
# Tăng model complexity:
transformer: d_model=128 (thay 64), num_layers=4 (thay 3)
gru: hidden_dim=256 (thay 128)
tcn: channels=[64, 128, 256, 256, 512] (thay 32, 64, 128, 128, 256)

# Hoặc tăng epochs:
epochs=200 (thay 150)
patience=20 (thay 15)
```

### 2️⃣ **Overfitting (val >> train)?**
```python
# Tăng regularization:
dropout=0.5 (thay 0.2/0.3)
noise_std=0.05 (thay 0.01)
weight_decay=1e-2 (thay 1e-3)

# Hoặc tăng batch_size:
batch_size=64 (thay 32)
```

### 3️⃣ **Training unstable (loss jumps)?**
```python
# Tăng gradient clipping:
torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # thay 1.0

# Hoặc tăng layer norm:
self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
```

### 4️⃣ **Model converge quá chậm?**
```python
# Tăng learning rate:
lr=5e-3 (thay 1e-3)

# Hoặc tăng scheduler:
T_max=100 (thay epochs)
```

---

## 💾 SAVING & LOADING BEST MODELS

```python
# Load tất cả 3 best models
transformer = Transformer(...).to(device)
transformer.load_state_dict(torch.load('best_transformer.pth', map_location=device, weights_only=True))

gru_seq2seq = GRUSeq2Seq(...).to(device)
gru_seq2seq.load_state_dict(torch.load('best_gru_s2s.pth', map_location=device, weights_only=True))

improved_tcn = ImprovedTCN(...).to(device)
improved_tcn.load_state_dict(torch.load('best_improved_tcn.pth', map_location=device, weights_only=True))

# Predict on new data
with torch.no_grad():
    X_new = torch.tensor(new_data_scaled).to(device).unsqueeze(0)
    
    pred_transformer = transformer(X_new).cpu().numpy()
    pred_gru = gru_seq2seq(X_new).cpu().numpy()
    pred_tcn = improved_tcn(X_new).cpu().numpy()
    
    ensemble_pred = (pred_transformer + pred_gru + pred_tcn) / 3
```

---

## ✅ CHECKLIST

- [ ] Run `optimized_models.py` successfully
- [ ] Check val loss decreases and plateaus
- [ ] Verify learning curves look healthy
- [ ] Check predictions visualizations
- [ ] Compare metrics - Ensemble should be best
- [ ] Try on real data (test1.xlsx + label1.xlsx)
- [ ] Verify NO data leakage (predictions should be reasonable, not perfect)

---

## 🎉 EXPECTED OUTCOMES

**Trước fix:**
- MSE: 0.75 - 33.15 (rất tệ)
- Predictions: Hoàn toàn sai (vì dùng ground truth)

**Sau fix (bộ 3 model mới):**
- MSE: ~0.3 - 0.8 (khá tốt)
- Predictions: Theo trend tương đối, không perfect (như nên)
- Ensemble: Tốt nhất của 3 model

---

**Chúc bạn huấn luyện thành công! 🚀**
