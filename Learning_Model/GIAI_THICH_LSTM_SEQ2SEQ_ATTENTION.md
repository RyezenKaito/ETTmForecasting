# 🔍 GIẢI THÍCH CHI TIẾT: Kiến Trúc Model Seq2SeqLSTM với Attention

## 📋 Tổng Quan

Model **Seq2SeqLSTM** là một kiến trúc **Encoder-Decoder** sử dụng **BiLSTM (Bidirectional LSTM)** kết hợp với **Scaled Dot-Product Attention** để dự báo chuỗi thời gian. Model này được thiết kế để:

- **Mã hóa (Encoding)**: Nắm bắt toàn bộ thông tin từ dữ liệu lịch sử
- **Giải mã (Decoding)**: Dự báo từng bước tiếp theo với sự chú ý (Attention) vào các phần quan trọng của dữ liệu lịch sử
- **Teacher Forcing**: Huấn luyện với giá trị thực tế, suy luận với dự báo của chính nó

---

## 1️⃣ THÀNH PHẦN: S2SEncoder (Encoder BiLSTM)

### Mã Code
```python
class S2SEncoder(nn.Module):
    def __init__(self, input_dim, hidden, n_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0, bidirectional=True)
    
    def forward(self, x):
        enc_out, (h, c) = self.lstm(x)
        h = torch.cat([h[0::2], h[1::2]], dim=2)
        c = torch.cat([c[0::2], c[1::2]], dim=2)
        return enc_out, h, c
```

### Giải Thích Chi Tiết

#### **Input**
- `x`: Tensor với shape `(batch_size, seq_len, input_dim)`
  - `batch_size`: số mẫu trong batch (e.g., 64)
  - `seq_len`: độ dài chuỗi lịch sử (e.g., 336 bước = 7 ngày)
  - `input_dim`: số features (e.g., 8: OT + 3 STL decomp + 4 time features)

#### **BiLSTM (Bidirectional LSTM)**

**"Bi" có nghĩa là gì?**
- LSTM chạy **2 lần**: một lần từ trái sang phải (forward), một lần từ phải sang trái (backward)
- Mỗi hướng có **hidden_size = 256** neurons

```
Forward:  →→→→→→→  (xử lý từ t=0 đến t=T-1)
Backward: ←←←←←←←  (xử lý từ t=T-1 về t=0)
```

**Tại sao BiLSTM tốt?**
- Forward LSTM: nắm bắt xu hướng từ quá khứ
- Backward LSTM: nắm bắt tương lai (context từ dữ liệu sau)
- Kết hợp: nhận được full context từ cả hai phía

#### **Output**

Hàm `forward()` trả về 3 thứ:

1. **`enc_out`**: shape `(batch_size, seq_len, hidden*2)`
   - Tất cả hidden states từ tất cả các bước thời gian
   - Mỗi phần tử là 512D (256 from forward + 256 from backward)
   - Dùng cho **Attention** để tính trọng số

2. **`h` (hidden state)**: shape `(batch_size, hidden*2)`
   - Trạng thái ẩn cuối cùng, sau khi xử lý toàn bộ chuỗi
   - Dùng để **khởi tạo trạng thái ban đầu của Decoder**

3. **`c` (cell state)**: shape `(batch_size, hidden*2)`
   - Trạng thái ô của LSTM (internal memory)
   - Cũng dùng để khởi tạo Decoder

#### **Ghép nối States từ 2 Hướng**

```python
h = torch.cat([h[0::2], h[1::2]], dim=2)
```

- `h[0::2]`: các hidden states từ lớp LSTM chẵn (forward)
- `h[1::2]`: các hidden states từ lớp LSTM lẻ (backward)
- `torch.cat(..., dim=2)`: concatenate theo dimension features

**Ví dụ với 2 layers:**
- Layer 0 (forward): h[0]
- Layer 1 (backward): h[1]
- Sau ghép: `[h[0], h[1]]` → shape (batch, 2, 512)

---

## 2️⃣ THÀNH PHẦN: S2SAttention (Scaled Dot-Product Attention)

### Mã Code
```python
class S2SAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.scale = (hidden * 2) ** -0.5  # 1/sqrt(hidden*2)
    
    def forward(self, h_last, enc_out):
        q = h_last.unsqueeze(1)  # (B, 1, hidden*2)
        w = torch.softmax(torch.bmm(q, enc_out.transpose(1, 2)) * self.scale, dim=-1)
        return torch.bmm(w, enc_out).squeeze(1)
```

### Giải Thích Chi Tiết

#### **Scaled Dot-Product Attention là gì?**

Attention tính **"độ liên quan"** giữa decoder state hiện tại và tất cả encoder states:

```
Attention Score = softmax(Q @ K^T / sqrt(d)) @ V
```

Ở đây:
- **Q (Query)**: hidden state cuối cùng của decoder → "đang tìm cái gì?"
- **K (Key)**: tất cả encoder states → "mỗi bước của input chứa gì?"
- **V (Value)**: tất cả encoder states → "lấy thông tin gì từ mỗi bước?"

#### **Bước 1: Tạo Query**

```python
q = h_last.unsqueeze(1)  # (B, hidden*2) → (B, 1, hidden*2)
```

- `h_last`: hidden state cuối cùng của decoder (current decoder state)
- `unsqueeze(1)`: thêm dimension thời gian để thành (batch, 1, hidden*2)

#### **Bước 2: Tính Attention Score**

```python
w = torch.softmax(torch.bmm(q, enc_out.transpose(1, 2)) * self.scale, dim=-1)
```

**Chi tiết:**
- `enc_out.transpose(1, 2)`: chuyển từ (B, seq_len, 512) → (B, 512, seq_len)
- `torch.bmm(q, enc_out.transpose(1, 2))`: 
  - Nhân ma trận batch: (B, 1, 512) × (B, 512, seq_len) = (B, 1, seq_len)
  - Kết quả: **score của mỗi bước input**
- `* self.scale`: chia cho `sqrt(512)` ≈ 0.044 để chuẩn hóa
  - Lý do: khi embedding dimension lớn, dot product sẽ rất lớn
  - Softmax của số lớn → gradient rất nhỏ (vanishing gradient)
  - Chia nhỏ scores → softmax output cân bằng hơn

- `softmax(..., dim=-1)`: áp dụng softmax trên seq_len
  - Kết quả: (B, 1, seq_len) với tổng mỗi hàng = 1
  - Ý nghĩa: **trọng số chú ý cho mỗi bước input**

```
Ví dụ:
Score: [0.1, 2.5, 0.3, 1.1, ...] (336 giá trị)
Scale: chia cho sqrt(512) → [-0.004, 0.11, -0.013, 0.048, ...]
Softmax: [0.002, 0.34, 0.001, 0.08, ...] (tổng = 1.0)
                    ↑ bước này được chú ý nhiều nhất!
```

#### **Bước 3: Áp Dụng Attention vào Value**

```python
return torch.bmm(w, enc_out).squeeze(1)
```

- `w`: (B, 1, seq_len) - trọng số chú ý
- `enc_out`: (B, seq_len, 512) - thông tin mỗi bước
- `torch.bmm(w, enc_out)`: (B, 1, seq_len) × (B, seq_len, 512) = (B, 1, 512)
  - **Context vector**: tổng có trọng số của tất cả encoder outputs
- `squeeze(1)`: bỏ dimension 1 → (B, 512)

**Ý nghĩa:**
- Nếu trọng số bước 10 = 0.5, bước 20 = 0.3, còn lại = 0.2
- Context = 0.5 × enc_out[10] + 0.3 × enc_out[20] + 0.2 × (các bước khác)
- **Decoder "tập trung" vào bước 10 và 20 vì chúng liên quan nhất**

#### **Hình Ảnh Attention**

```
Encoder Output:        Attention Weight:        Context Vector:
[x₁ 512D]              [0.02]                    
[x₂ 512D]              [0.34] ← cao nhất   →    weighted sum
[x₃ 512D]              [0.15]                    = context (512D)
[...  ]                [...]
[x₃₃₆ 512D]            [0.01]

Attention "chọn" từng phần của mỗi encoder state
dựa trên mức độ liên quan với decoder state hiện tại
```

---

## 3️⃣ THÀNH PHẦN: S2SDecoder (Decoder LSTM with Attention)

### Mã Code
```python
class S2SDecoder(nn.Module):
    def __init__(self, dec_in, hidden, n_layers, dropout):
        super().__init__()
        self.attn = S2SAttention(hidden)
        self.lstm = nn.LSTM(dec_in + hidden * 2, hidden * 2, n_layers, 
                            batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Sequential(nn.LayerNorm(hidden * 2),
                                nn.Linear(hidden * 2, 128), nn.GELU(), 
                                nn.Dropout(dropout),
                                nn.Linear(128, 1))
    
    def forward(self, dec_in, h, c, enc_out):
        ctx = self.attn(h[-1], enc_out)
        inp = torch.cat([dec_in.squeeze(1), ctx], dim=-1).unsqueeze(1)
        out, (h, c) = self.lstm(inp, (h, c))
        return self.fc(out.squeeze(1)), h, c
```

### Giải Thích Chi Tiết

#### **Input của Forward**

1. **`dec_in`**: shape `(B, 1, dec_in_dim)` 
   - Giá trị input cho bước decode hiện tại
   - Chứa: [previous_output (1D), time_features (4D)] = 5D tổng

2. **`h, c`**: trạng thái LSTM từ Encoder
   - Khởi tạo trạng thái bên Decoder

3. **`enc_out`**: tất cả encoder outputs (B, seq_len, 512)
   - Dùng cho Attention

#### **Bước 1: Tính Attention Context**

```python
ctx = self.attn(h[-1], enc_out)  # (B, 512)
```

- `h[-1]`: hidden state cuối cùng từ trạng thái hiện tại của decoder
- `ctx`: context vector từ Attention
- Shape: (batch_size, 512)

#### **Bước 2: Concatenate Input + Context**

```python
inp = torch.cat([dec_in.squeeze(1), ctx], dim=-1).unsqueeze(1)
```

- `dec_in.squeeze(1)`: (B, 1, 5) → (B, 5) - loại bỏ dimension 1
- `ctx`: (B, 512)
- `torch.cat(..., dim=-1)`: (B, 5+512) = (B, 517)
- `unsqueeze(1)`: (B, 517) → (B, 1, 517) - thêm dimension thời gian

**Ý nghĩa:**
- Decoder nhận **đầu vào hiện tại** (previous prediction + time features)
- CỘNG với **context từ input** (attention over encoder)
- Sự kết hợp này giúp decoder "biết" cần chú ý đến đâu

#### **Bước 3: Forward qua LSTM**

```python
out, (h, c) = self.lstm(inp, (h, c))
```

- Input: (B, 1, 517)
- Initial state: (h, c) từ trạng thái trước
- Output: (B, 1, 512)
- **LSTM cập nhật trạng thái** h, c cho bước kế tiếp

#### **Bước 4: Output Linear Layer**

```python
return self.fc(out.squeeze(1)), h, c
```

```python
self.fc = nn.Sequential(
    nn.LayerNorm(512),                    # Layer norm (chuẩn hóa)
    nn.Linear(512, 128), nn.GELU(),       # Lớp ẩn
    nn.Dropout(0.2),
    nn.Linear(128, 1)                     # Output layer (dự báo 1 giá trị)
)
```

- Input: (B, 512)
- LayerNorm: chuẩn hóa features
- Linear → 128 neurons: capture non-linear relationships
- GELU activation: smooth activation function
- Dropout: regularization
- Linear → 1 neuron: **dự báo 1 giá trị (OT tại bước tiếp theo)**
- Output: (B, 1)

---

## 4️⃣ THÀNH PHẦN: Seq2SeqLSTM (Mô Hình Chính)

### Mã Code
```python
class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_dim, hidden=256, n_layers=2, dropout=0.2,
                 dec_in_dim=5, pred_len=24, target_index=0):
        super().__init__()
        self.pred_len     = pred_len
        self.target_index = target_index
        self.encoder      = S2SEncoder(input_dim, hidden, n_layers, dropout)
        self.decoder      = S2SDecoder(dec_in_dim, hidden, n_layers, dropout)

    def forward(self, x, y=None, future_cov=None, tf_ratio=0.5):
        enc_out, h, c = self.encoder(x)
        prev_out = x[:, -1, self.target_index].unsqueeze(-1)  # (B, 1)
        outputs  = []
        
        for t in range(self.pred_len):
            cov_t  = future_cov[:, t, :] if future_cov is not None else None
            dec_in = torch.cat([prev_out, cov_t], dim=-1).unsqueeze(1) \
                     if cov_t is not None else prev_out.unsqueeze(1)
            pred, h, c = self.decoder(dec_in, h, c, enc_out)
            outputs.append(pred)
            
            # Teacher Forcing
            use_tf   = self.training and y is not None and random.random() < tf_ratio
            prev_out = y[:, t].unsqueeze(-1) if use_tf else pred
        
        return torch.cat(outputs, dim=1)  # (B, pred_len)
```

### Giải Thích Chi Tiết

#### **Khởi Tạo (Init)**

```python
self.encoder = S2SEncoder(input_dim, hidden, n_layers, dropout)
self.decoder = S2SDecoder(dec_in_dim, hidden, n_layers, dropout)
```

- **Encoder**: chuyên xử lý input
- **Decoder**: chuyên dự báo output từng bước

#### **Forward: Encoding Phase**

```python
enc_out, h, c = self.encoder(x)
```

- Encode toàn bộ input sequence (336 bước)
- Nhận ra mọi thông tin cần thiết từ lịch sử
- `enc_out`: (B, 336, 512) - tất cả encoder states (dùng cho Attention)
- `h, c`: (B, 512) - trạng thái cuối (dùng khởi tạo Decoder)

#### **Forward: Decoding Phase - Autoregressive Loop**

```python
prev_out = x[:, -1, self.target_index].unsqueeze(-1)  # (B, 1)
outputs  = []

for t in range(self.pred_len):  # self.pred_len = 24
    # ...
```

**Bước 1: Lấy giá trị khởi đầu**
- `x[:, -1, self.target_index]`: lấy **giá trị cuối cùng** của input target (OT)
- Shape: (B,) → unsqueeze: (B, 1)
- Đây là "previous output" để bắt đầu dự báo

**Bước 2: Chuẩn bị input cho Decoder**

```python
cov_t = future_cov[:, t, :] if future_cov is not None else None
dec_in = torch.cat([prev_out, cov_t], dim=-1).unsqueeze(1)
```

- `future_cov[:, t, :]`: lấy time features của bước t
  - Shape: (B, 4) - [time_sin, time_cos, day_sin, day_cos]
- `torch.cat([prev_out, cov_t], dim=-1)`: (B, 1) + (B, 4) = (B, 5)
- `unsqueeze(1)`: (B, 5) → (B, 1, 5)
- **Dec_in chứa**: [previous output + time features]

**Bước 3: Decode một bước**

```python
pred, h, c = self.decoder(dec_in, h, c, enc_out)
outputs.append(pred)
```

- Decoder nhận: (B, 1, 5) input + (B, 512) hidden state + encoder outputs
- Trả về: (B, 1) prediction
- **Lưu vào outputs list**

**Bước 4: Teacher Forcing**

```python
use_tf = self.training and y is not None and random.random() < tf_ratio
prev_out = y[:, t].unsqueeze(-1) if use_tf else pred
```

- Nếu đang **training** (`self.training=True`)
  - và có **ground truth** (`y is not None`)
  - và **random < tf_ratio** (probability decay scheduling)
- Thì dùng **giá trị thực tế** (`y[:, t]`) làm input cho bước tiếp theo
- Nếu không, dùng **dự báo của model** (`pred`)

**Tại sao Teacher Forcing?**

```
Huấn luyện (Training):
- Bước 1: dự báo t+1 → có thể sai
  ↓
- Bước 2: nếu không có TF, input = dự báo sai từ bước 1 → error accumulate
  ↓
- Kết quả: model học từ sai lầm của chính nó

Với Teacher Forcing:
- Bước 1: dự báo t+1 → có thể sai
  ↓
- Bước 2: input = giá trị thực tế t+1 (từ ground truth)
  ↓
- Kết quả: model học từ đúng, convergence nhanh hơn
```

**Scheduled Decay:**
- Đầu training: `tf_ratio=0.6` → 60% xác suất dùng ground truth
- Cuối training: `tf_ratio=0.1` → 10% xác suất dùng ground truth
- Lý do: từ từ phụ thuộc vào dự báo của chính nó (exposure bias mitigation)

#### **Forward: Kết Hợp Output**

```python
return torch.cat(outputs, dim=1)  # (B, pred_len)
```

- `outputs`: list của 24 tensors, mỗi (B, 1)
- `torch.cat(..., dim=1)`: (B, 1) + (B, 1) + ... = (B, 24)
- **Kết quả cuối cùng**: (batch_size, 24) = dự báo 24 bước

---

## 5️⃣ FLOW ĐẦY ĐỦ: Input → Output

### Ví Dụ Cụ Thể

```
Input:
  x shape:          (64, 336, 8)   ← 64 batches, 336 timesteps, 8 features
  y shape:          (64, 24)       ← ground truth dự báo (train only)
  future_cov shape: (64, 24, 4)    ← time features for 24 future steps

Step 1: ENCODING
  ──────────────
  x (64, 336, 8)
    ↓ BiLSTM forward
    ↓ BiLSTM backward
  enc_out (64, 336, 512) ← tất cả encoder states
  h (64, 512)            ← cuối state (forward direction)
  c (64, 512)            ← cuối cell state

Step 2: DECODING - Loop qua 24 bước
  ──────────────────────────────────
  
  Khởi tạo:
    prev_out = x[:, -1, 0] = (64,) → unsqueeze → (64, 1)
    outputs = []
  
  Vòng lặp t=0:
    ├─ Attention: 
    │   ├─ query = h[-1] (64, 512)
    │   ├─ score = softmax(query @ enc_out^T / sqrt(512))  (64, 336)
    │   └─ context = score @ enc_out  (64, 512)
    ├─ Concatenate:
    │   ├─ prev_out (64, 1)
    │   ├─ future_cov[:, 0, :] (64, 4)
    │   └─ dec_in = cat → (64, 5) → unsqueeze → (64, 1, 5)
    ├─ LSTM forward:
    │   ├─ input: dec_in + context = (64, 1, 517)
    │   ├─ h, c từ trạng thái trước
    │   └─ output (64, 1, 512)
    ├─ Dense head:
    │   ├─ LayerNorm → Linear(512→128) → GELU → Linear(128→1)
    │   └─ pred (64, 1) ← dự báo bước 0
    ├─ Teacher Forcing:
    │   ├─ random(0,1) < 0.6? → Có (first epoch)
    │   └─ prev_out = y[:, 0] ← từ ground truth
    └─ outputs.append(pred)
  
  Vòng lặp t=1:
    ├─ Attention: (giống như trên)
    ├─ Concatenate:
    │   ├─ prev_out = y[:, 0] (từ TF bước trước) (64, 1)
    │   ├─ future_cov[:, 1, :] (64, 4)
    │   └─ dec_in (64, 1, 5)
    ├─ LSTM forward: (update h, c)
    ├─ Dense: pred (64, 1) ← dự báo bước 1
    └─ outputs.append(pred)
  
  ... (lặp 22 lần nữa)
  
  Vòng lặp t=23:
    ├─ (tương tự)
    └─ outputs.append(pred) ← dự báo bước 23

Step 3: CONCATENATE OUTPUTS
  ──────────────────────────
  outputs: [(64,1), (64,1), ..., (64,1)]  ← 24 tensors
  result = torch.cat(outputs, dim=1)
  result.shape = (64, 24) ← dự báo cuối cùng
```

---

## 6️⃣ ATTENTION VISUALIZATION

### Cách Hoạt Động

```
Tại bước decode t=5 (dự báo 5 bước vào tương lai):

Decoder state: h[-1] (512D) → Attention

Encoder states: enc_out (336, 512D)
  x₁: [0.1, -0.2, 0.5, ..., 0.3]
  x₂: [0.3, 0.1, -0.1, ..., 0.2]
  ...
  x₃₃₆: [-0.2, 0.4, 0.2, ..., 0.1]

Attention mechanism:
  Query: h[-1] (decoder state hiện tại)
  
  Đối với mỗi encoder state xᵢ:
    score_i = h[-1] · xᵢ / sqrt(512)
  
  scores = [2.3, 0.5, 5.2, 1.1, ..., 0.3]  ← 336 scores
  
  weights = softmax(scores)
  weights = [0.01, 0.001, 0.45, 0.02, ..., 0.0001]  ← tổng = 1.0
  
  context = Σ(weight_i × enc_out_i)
          = 0.01×x₁ + 0.001×x₂ + 0.45×x₃ + 0.02×x₄ + ... + 0.0001×x₃₃₆
  
→ Decoder "tập trung" 45% vào bước 3 của input!
→ Điều này giúp decoder "biết" phần nào của input quan trọng nhất
```

### Ví Dụ: Tại Sao Attention Hữu Ích?

```
Scenario: Dự báo nhiệt độ OT

Input sequence: [t₁=20°C, t₂=21°C, ..., t₃₃₆=22°C]
                 ↑ hôm qua          ↑ bây giờ

Dự báo bước 1 (1 giờ tiếp theo):
  - Attention có thể tập trung vào t₃₃₀-t₃₃₆ (những giờ gần đây nhất)
  - Context ≈ [21, 21.5, 21.8, 22, ...] (gần đây)
  - Dự báo: ≈ 22.2°C (xu hướng tăng nhẹ)

Dự báo bước 12 (12 giờ tiếp theo):
  - Attention có thể tập trung vào t₂₄₀-t₂₈₀ (cùng thời điểm hôm qua)
  - Context ≈ [19, 18.5, 18.8, 19] (pattern 24h trước)
  - Dự báo: ≈ 18.5°C (dựa vào tính chu kỳ hàng ngày)

→ Attention tự động "học" tập trung vào các phần khác nhau
  tùy vào lúc cần dự báo!
```

---

## 7️⃣ HYPERPARAMETER CHÍNH

Từ notebook:

```python
# Model Architecture
S2S_HIDEN_SIZE   = 256        # 512D sau concatenate forward/backward
S2S_NUMER_LAYERS = 2          # 2 LSTM layers
S2S_DROPOUT      = 0.2        # Regularization

# Training
DEC_IN_DIM       = 5          # 1 (prev output) + 4 (time features)
S2S_LR           = 1E-4       # Learning rate
S2S_WEIGHT       = 1E-3       # L2 regularization
S2S_ETA_MIN      = 1E-6       # Min LR for cosine annealing

# Decoding
pred_len         = 24         # Dự báo 24 bước
tf_ratio         = 0.6 ~ 0.1  # Teacher forcing: decay 60% → 10%
```

### Số Lượng Parameter

```python
Seq2Seq params: ~1,256,385

Breakdown:
├─ Encoder BiLSTM: 
│  ├─ Input 8D → 256D forward LSTM
│  ├─ Input 8D → 256D backward LSTM
│  ├─ 2 layers → 2× parameters
│  └─ ≈ 640k params
│
├─ Attention: 
│  └─ ≈ 5k params
│
└─ Decoder:
   ├─ LSTM: 517D input (5 dec_in + 512 context) → 512D
   ├─ 2 layers
   ├─ Dense head: 512→128→1
   └─ ≈ 611k params
```

---

## 8️⃣ TRAINING DETAILS

### Loss Function

```python
loss = MSE_loss(pred, y_true) + TREND_LAMBDA × trend_loss(pred, y_true)

Trong đó:
  MSE_loss: L2 loss trên giá trị dự báo
  
  trend_loss: MSE trên difference
    diff_pred = pred[:, 1:] - pred[:, :-1]  ← gradient
    diff_true = y_true[:, 1:] - y_true[:, :-1]
    trend_loss = MSE(diff_pred, diff_true)
    
  TREND_LAMBDA = 0.3
    → Balance: 70% value loss + 30% trend loss
```

**Tại sao trend loss?**
- Model học không chỉ dự báo giá trị đúng
- Mà cả **tốc độ thay đổi** (slope) cũng phải đúng
- Ngăn chặn dự báo "bằng phẳng" hoặc "dao động"

### Learning Rate Schedule

```python
optimizer = AdamW(lr=1e-4, weight_decay=1e-3)
scheduler = CosineAnnealingLR(T_max=100, eta_min=1e-6)

Learning rate giảm từ 1e-4 → 1e-6 theo cosine curve:
  lr(t) = eta_min + 0.5×(eta_max - eta_min) × (1 + cos(π×t/T_max))
  
Epoch 1:    lr ≈ 1.0e-4
Epoch 50:   lr ≈ 5.0e-5
Epoch 100:  lr ≈ 1.0e-6
```

---

## 9️⃣ COMPARISION: Seq2Seq vs TCN

| Aspect | Seq2Seq + Attention | TCN |
|--------|-------------------|-----|
| **Kiến trúc** | RNN (LSTM) | CNN |
| **Receptive Field** | Tất cả (thông qua recurrence) | Hạn chế (dilation) |
| **Attention** | Có ✓ | Không |
| **Sequential** | Buộc phải dự báo từng bước | Có thể dự báo cùng lúc |
| **Điểm mạnh** | Nắm bắt dependencies dài hạn, Attention giúp hiểu được quan trọng ở đâu | Nhanh, Có thể parallelizable, Đơn giản |
| **Điểm yếu** | Chậm (sequential), Khó train (vanishing gradient) | Receptive field hạn chế, Không có built-in attention |

---

## 🔟 TÓM TẮT FLOW

```
┌─────────────────────────────────────────────────────────────┐
│                    FULL FORWARD PASS                        │
└─────────────────────────────────────────────────────────────┘

INPUT (B, 336, 8):  ←─ lịch sử 7 ngày

  ↓ BiLSTM Encoder (forward + backward)
  ↓
ENC_OUT (B, 336, 512):  ←─ tất cả encoder states
H, C (B, 512):           ←─ cuối state
  
  ↓ Autoregressive Decoder Loop (24 bước)
  ├─ t=0: prev_out (B, 1) → Attention → Context → LSTM → pred (B, 1)
  │                         ↑ 336 →  ↓ 512     ↑ 512→1
  ├─ t=1: prev_out (B, 1) → Attention → Context → LSTM → pred (B, 1)
  │
  └─ t=23: prev_out (B, 1) → Attention → Context → LSTM → pred (B, 1)
  
  ↓ Concatenate
  ↓
OUTPUT (B, 24):  ←─ dự báo 24 giờ tiếp theo

LOSS = MSE(output, y_true) + 0.3×trend_loss(output, y_true)
```

---

## ✅ KẾT LUẬN

**Seq2SeqLSTM với Attention** là một mô hình mạnh mẽ cho dự báo chuỗi thời gian bởi vì:

1. **BiLSTM Encoder**: Nắm bắt full context từ cả hai hướng
2. **Scaled Dot-Product Attention**: Model học được tập trung vào những phần quan trọng của input
3. **Autoregressive Decoding**: Dự báo từng bước một, conditioned on previous predictions
4. **Teacher Forcing**: Giúp training convergence nhanh hơn
5. **Time Features**: Chứa thông tin chu kỳ (sin/cos of hour, day of week)

Model này rất phù hợp cho:
- Time series forecasting (dự báo chuỗi thời gian)
- Sequence-to-sequence tasks
- Các trường hợp cần interpretability (xem Attention weights)

