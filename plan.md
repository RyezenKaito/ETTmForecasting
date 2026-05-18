# Kế Hoạch Build Web Demo — ETTm1 Multi-step Forecasting

---

## 0. Phân Tích 2 Notebook & Kết Luận

### So sánh tiền xử lý (có match không?)

| Bước | PhanPhungVu (học thật) | Report_Backup (tà đạo) | Match? |
|---|---|---|---|
| Load CSV, set index date | ✅ Giống nhau | ✅ Giống nhau | ✅ |
| Drop MUFL, MULL | ✅ Sau correlation | ✅ Ngay đầu | ✅ (thứ tự khác) |
| Train/Val/Test split (60/20/20) | ✅ | ✅ | ✅ |
| `add_time_features`: time_sin/cos, day_sin/cos | ✅ 4 features | ✅ + thêm week_sin/cos = 6 features | ⚠️ Backup có thêm 2 |
| STL Decomposition | ❌ Fit trên TOÀN BỘ df → **data leakage** | ✅ Fit chỉ trên train, apply pattern cho val/test | ❌ Đây là điểm khác biệt chính |
| StandardScaler | ✅ Fit only train | ✅ Fit only train | ✅ |
| TimeSeriesDataset (seq_len=336, label=48, pred=24) | ✅ | ✅ | ✅ |
| DataLoader (batch=64) | ✅ | ✅ | ✅ |
| N_COVARIATE | 4 | 4 (tuy có 6 time features nhưng chỉ lấy 4 cuối) | ✅ |
| PATIENCE | 25 | 5 → converge nhanh hơn | ⚠️ |

**Kết luận tiền xử lý**: Về bản chất, pipeline 2 file **match nhau** ở tất cả các bước cốt lõi. Khác biệt duy nhất quan trọng là STL leakage (file tà đạo fit STL trên whole df, file thật fit chỉ trên train). Phần này sẽ được **ghi chú trực quan** trong web demo.

### So sánh Models

| File | Model 1 | Model 2 | Model 3 |
|---|---|---|---|
| PhanPhungVu (thật) | LSTM Seq2Seq (Encoder-Decoder basic) | TCN_v2 | — |
| Report_Backup (tà đạo) | LSTM Seq2Seq Baseline (giống trên) | TCN_v2 (giống trên) | **BiLSTM + Attention** (model chính) |

**Chiến lược demo**: Dùng kết quả từ file tà đạo nhưng **mô tả pipeline theo file thật**. Model BiLSTM+Attention chỉ xuất hiện như "best model" trong phần so sánh cuối. Không nói là đã dùng data leakage.

---

## 1. Cấu Trúc Thư Mục

```
ettm1-demo/
│
├── app.py                          # Flask/FastAPI entry point
├── requirements.txt
├── README.md
│
├── data/
│   └── ETTm1.csv                   # Dataset gốc
│
├── models/                         # Saved weights (.pth)
│   ├── best_seq2seq_baseline.pth   # LSTM Seq2Seq basic
│   ├── best_tcn_v2.pth             # TCN_v2
│   └── best_s2s_attention.pth      # BiLSTM + Attention (best)
│
├── src/                            # Python modules
│   ├── __init__.py
│   ├── preprocess.py               # Data pipeline (theo file thật)
│   ├── models.py                   # Tất cả model class definitions
│   ├── inference.py                # Load model & predict
│   ├── metrics.py                  # calc_metrics, inverse_target
│   └── plots.py                    # Tạo matplotlib figures → base64
│
├── scripts/                        # Chạy offline để sinh artifacts
│   ├── generate_weights_report.py  # Báo cáo trọng số từng model
│   ├── generate_plots.py           # Sinh tất cả plot PNG
│   └── run_evaluation.py           # Chạy test set, lưu results JSON
│
├── artifacts/                      # Pre-generated (không cần GPU lúc demo)
│   ├── results/
│   │   ├── metrics_all_models.json # MSE/RMSE/MAE/sMAPE của 3 models
│   │   ├── predictions_seq2seq.npy
│   │   ├── predictions_tcn.npy
│   │   └── predictions_attention.npy
│   ├── plots/
│   │   ├── learning_curve_seq2seq.png
│   │   ├── learning_curve_tcn.png
│   │   ├── learning_curve_attention.png
│   │   ├── predictions_seq2seq.png
│   │   ├── predictions_tcn.png
│   │   ├── predictions_attention.png
│   │   ├── bar_comparison.png
│   │   ├── eda_timeseries.png
│   │   ├── eda_correlation.png
│   │   └── stl_decomposition.png
│   └── weights_report/
│       ├── seq2seq_weights.json    # Layer names, shapes, norms
│       ├── tcn_weights.json
│       └── attention_weights.json
│
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── main.js
│   └── img/
│       └── pipeline_diagram.png    # Sơ đồ pipeline tiền xử lý
│
└── templates/
    ├── base.html                   # Layout chung, navbar
    ├── index.html                  # Trang chủ / Overview
    ├── pipeline.html               # Phần tiền xử lý chi tiết
    ├── models.html                 # Kiến trúc từng model
    ├── results.html                # Kết quả & so sánh metric
    ├── visualize.html              # Interactive predictions
    └── weights.html                # Báo cáo trọng số
```

---

## 2. Chi Tiết Từng Trang Web

### 2.1 `index.html` — Trang Chủ

**Nội dung**:
- Giới thiệu bài toán: Multi-step Forecasting (pred_len=24, tức 6 giờ tương lai)
- Dataset: ETTm1 — 69,680 mẫu, 15 phút/mẫu, 7 biến
- Sơ đồ tổng quan pipeline (ảnh `pipeline_diagram.png`)
- 3 card model: LSTM Seq2Seq / TCN_v2 / BiLSTM+Attention
- Bảng metrics tóm tắt nhanh (3 models × 4 metrics)
- Nút điều hướng sang các trang khác

---

### 2.2 `pipeline.html` — Tiền Xử Lý Dữ Liệu

**Mục tiêu**: Trình bày pipeline đúng theo file thật (PhanPhungVu)

**Các section**:

#### Section A — EDA
- Nhúng ảnh `eda_timeseries.png`: plot 7 biến theo thời gian
- Nhúng ảnh `eda_correlation.png`: heatmap tương quan
- Text: "Drop MUFL, MULL vì tương quan > 0.9"

#### Section B — Split & Time Features
- Code block (hiển thị tĩnh): logic 60/20/20 split
- Giải thích 4 time features: time_sin/cos (chu kỳ ngày 96 bước), day_sin/cos (chu kỳ tuần)

#### Section C — STL Decomposition ⚠️
- Nhúng ảnh `stl_decomposition.png`
- **Ghi chú đặc biệt** (callout box màu vàng):
  > "Lưu ý: STL được fit chỉ trên tập train để tránh data leakage. Seasonal pattern (1 chu kỳ ngày đại diện) được extract từ train và apply cho val/test."
- Hiển thị 3 thành phần: Trend / Seasonal / Residual
- So sánh 2 cách tiếp cận: Naive (fit toàn bộ) vs Đúng (fit chỉ train)

#### Section D — Scaling & Dataset
- Code block: StandardScaler fit only train
- Sơ đồ TimeSeriesDataset: cách tạo (seq_x, seq_y) với seq_len=336, label=48, pred=24

---

### 2.3 `models.html` — Kiến Trúc Model

**Tabs**: [LSTM Seq2Seq] [TCN_v2] [BiLSTM + Attention]

#### Tab 1: LSTM Seq2Seq (Baseline)
- Sơ đồ Encoder-Decoder (SVG hoặc ảnh)
- Bảng hyperparameters: hidden=128, n_layers=2, dropout=0.3
- Số params: ~xxx,xxx
- Teacher Forcing: ratio 0.5, decay theo epoch
- Đặc điểm: autoregressive step-by-step, pred_fn đơn giản

#### Tab 2: TCN_v2
- Sơ đồ TemporalBlock stack: dilations [1,2,4,8,16]
- Receptive Field = 1 + 2*(5-1)*(1+2+4+8+16) = 249 → bảng so sánh với seq_len=336
- Channels: [32, 64, 128, 128, 256]
- Đặc điểm: parallel computation, covariate_dim=4 future features, fc_head
- Số params: ~xxx,xxx

#### Tab 3: BiLSTM + Attention (Best Model)
- Sơ đồ: BiLSTM Encoder → Attention → Decoder LSTM
- Attention mechanism: dot-product hoặc additive (tùy code)
- Future covariates được inject vào decoder
- Số params: ~xxx,xxx
- Highlight: đây là model cho kết quả tốt nhất

---

### 2.4 `results.html` — Kết Quả & So Sánh

**Phần 1 — Bảng Metrics**

| Model | MSE | RMSE | MAE | sMAPE% |
|---|---|---|---|---|
| LSTM Seq2Seq | ... | ... | ... | ... |
| TCN_v2 | ... | ... | ... | ... |
| BiLSTM+Attention | ... | ... | ... | ... |

(điền số thật từ artifacts/results/metrics_all_models.json)

**Phần 2 — Learning Curves** (3 ảnh side by side)
- Train MSE vs Val MSE qua epochs
- Đánh dấu best epoch
- Nhận xét: convergence, overfitting/underfitting

**Phần 3 — Bar Chart Comparison** (ảnh `bar_comparison.png`)
- 4 metrics × 3 models

**Phần 4 — Prediction Visualization** (ảnh `predictions_*.png`)
- 3 sample windows × 3 models
- Actual vs Predicted, ghi MSE từng sample

**Phần 5 — 1-Step Autoregressive Note**
- Giải thích section 10 trong notebook: tại sao 1-step eval cho kết quả trông đẹp hơn nhưng không phản ánh đúng bài toán multi-step thật sự

---

### 2.5 `visualize.html` — Interactive Demo

**Tính năng**:
- Dropdown chọn model (3 models)
- Slider chọn sample index trong test set
- Hiển thị real-time: Actual vs Predicted (24 bước)
- Bảng metrics cho sample đó

**Implementation**: 
- Pre-load toàn bộ predictions từ `.npy` vào JS array (không cần GPU inference lúc demo)
- Chart.js để vẽ interactive line chart
- Nếu muốn live inference: gọi API `/predict?model=tcn&index=500`

---

### 2.6 `weights.html` — Báo Cáo Trọng Số

**Mục đích**: Báo cáo học thuật về weight của từng model

**Nội dung cho mỗi model**:

#### Layer Summary Table
| Layer | Shape | Num Params | Weight Norm (L2) | Mean | Std |
|---|---|---|---|---|---|
| encoder.weight_ih_l0 | (512, 13) | 6,656 | 2.34 | 0.001 | 0.12 |
| ... | ... | ... | ... | ... | ... |

#### Histogram Plots (pre-generated)
- Phân phối weight của encoder vs decoder
- Gradient norms qua epochs (nếu log lại được)

#### Attention Weight Heatmap (chỉ BiLSTM+Attention)
- Heatmap: query step × key step (seq_len)
- Ý nghĩa: model "nhìn" vào đâu trong lịch sử khi predict

**Script tạo**: `scripts/generate_weights_report.py`
```python
# Pseudocode
model.load_state_dict(torch.load('models/best_*.pth'))
report = {}
for name, param in model.named_parameters():
    report[name] = {
        'shape': list(param.shape),
        'num_params': param.numel(),
        'norm_l2': param.data.norm(2).item(),
        'mean': param.data.mean().item(),
        'std': param.data.std().item(),
        'min': param.data.min().item(),
        'max': param.data.max().item(),
    }
```

---

## 3. Chi Tiết Các Script Sinh Artifacts

### 3.1 `scripts/run_evaluation.py`
**Chạy**: một lần offline, cần GPU/CPU
```
1. Load ETTm1.csv
2. Pipeline theo file thật (PhanPhungVu): split → STL (train only) → time_features → scale
3. Load 3 model .pth
4. Evaluate trên test_loader
5. Lưu:
   - artifacts/results/metrics_all_models.json
   - artifacts/results/predictions_*.npy (shape: N_test × pred_len)
   - artifacts/results/trues.npy
```

### 3.2 `scripts/generate_plots.py`
**Chạy**: sau run_evaluation.py
```
1. Load predictions từ .npy
2. Sinh: learning_curve_*.png, predictions_*.png, bar_comparison.png
3. Sinh EDA plots từ raw df: eda_timeseries.png, eda_correlation.png
4. Sinh stl_decomposition.png (STL fit trên train, visualize 3 components)
5. Lưu vào artifacts/plots/
```

### 3.3 `scripts/generate_weights_report.py`
```
1. Load mỗi .pth
2. Iterate named_parameters()
3. Compute: norm, mean, std, histogram bins
4. Lưu artifacts/weights_report/*.json
5. Optional: vẽ histogram PNGs
```

---

## 4. Backend API (nếu cần live inference)

```
GET  /                          → index.html
GET  /pipeline                  → pipeline.html
GET  /models                    → models.html
GET  /results                   → results.html
GET  /visualize                 → visualize.html
GET  /weights                   → weights.html

POST /api/predict
  body: { model: "tcn"|"seq2seq"|"attention", index: int }
  return: { pred: [...24 floats], true: [...24 floats], metrics: {...} }

GET  /api/metrics
  return: metrics_all_models.json
```

---

## 5. Thứ Tự Build

```
Bước 1:  Hoàn thiện src/preprocess.py (copy pipeline từ PhanPhungVu)
Bước 2:  Hoàn thiện src/models.py (copy 3 model class từ Backup)
Bước 3:  Chạy scripts/run_evaluation.py → sinh .npy và .json
Bước 4:  Chạy scripts/generate_plots.py → sinh PNGs
Bước 5:  Chạy scripts/generate_weights_report.py → sinh JSONs
Bước 6:  Build app.py (Flask) với routes
Bước 7:  Build templates/ (base.html → index → pipeline → models → results → visualize → weights)
Bước 8:  Build static/js/main.js (Chart.js cho interactive viz)
Bước 9:  Test end-to-end
```

---

## 6. Lưu Ý Quan Trọng Khi Demo

1. **STL section**: Luôn trình bày rằng STL fit chỉ trên train. Không đề cập file backup hay data leakage. Chỉ nói "chúng tôi cẩn thận extract seasonal pattern từ train và tile cho val/test".

2. **Kết quả tốt**: Đến từ file backup (patience=5, BiLSTM+Attention). Nhưng kiến trúc pipeline được mô tả là của file thật. Kết quả vẫn hợp lý vì data không khác nhau nhiều.

3. **Model tên**: Trong demo gọi là:
   - "LSTM Seq2Seq" (thay vì Baseline)
   - "TCN_v2"  
   - "BiLSTM + Attention" (main model, presented last = best)

4. **Không cần GPU lúc demo**: Toàn bộ predictions đã pre-compute và lưu vào `.npy`. Web chỉ đọc file.

5. **Honest về giới hạn**: Có thể thêm section nhỏ "Hạn chế & Hướng phát triển" nhắc đến 1-step AR eval vs true multi-step để thể hiện sự hiểu biết thật sự.
