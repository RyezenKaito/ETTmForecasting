# ETT Temperature Forecasting

Dự báo dầu biến áp (Oil Temperature — OT) trên dataset **ETTm1** sử dụng hai kiến trúc:

| Mô hình | Loại | Mô tả |
|---|---|---|
| **Seq2SeqLSTM** | Baseline | Bidirectional Encoder + Scaled Dot-Product Attention Decoder |
| **Informer** | Advanced | ProbSparse Self-Attention, O(L log L) complexity |

---

## Cấu trúc project

```
ETTForecasting/
├── data/
│   └── ETTm1.csv
├── checkpoints/           ← model weights (auto-created)
├── results/               ← plots & metrics (auto-created)
├── src/
│   ├── data/
│   │   ├── preprocessing.py   ← STL decomp, Winsorize, StandardScaler
│   │   └── dataset.py         ← TimeSeriesDataset4Seq, ETTDatasetInformer
│   ├── models/
│   │   ├── seq2seq.py         ← Encoder / Attention / Decoder / Seq2SeqLSTM
│   │   └── informer/          ← Informer2020 (adapted)
│   └── utils/
│       ├── metrics.py         ← MAE, RMSE, MAPE, MSE
│       └── tools.py           ← EarlyStopping, plot_history
├── config.py              ← ALL hyperparameters in one place
├── train_seq2seq.py       ← Train Seq2Seq
├── train_informer.py      ← Train Informer
├── evaluate.py            ← Compare both models on test set
└── requirements.txt
```

---

## Cài đặt

```bash
pip install -r requirements.txt
```

---

## Cách chạy

### 1. Train Seq2SeqLSTM
```bash
python train_seq2seq.py
```

### 2. Train Informer
```bash
python train_informer.py
```

### 3. So sánh kết quả
```bash
python evaluate.py
```

---

## Thông số kỹ thuật

### Dataset
- **File**: `data/ETTm1.csv`  (ETT Electricity Transformer Temperature, 15-min interval)
- **Target**: `OT` (Oil Temperature)
- **Features** sau preprocessing: **10 cột**
  - `HUFL, HULL, LUFL, LULL` (raw signals × 4 — ETTm1 có 4 load features)
  - `trend, seasonal, residual` (STL decomposition của OT)
  - `time_sin, time_cos` (cyclical time encoding)
  - `OT` (target)
- **Split**: Train 60% / Val 20% / Test 20% (chronological)

### Sequence lengths
| Parameter | Value |
|---|---|
| `SEQ_LEN`   | 336 steps (~3.5 ngày) |
| `LABEL_LEN` | 168 steps (= SEQ_LEN // 2, Informer decoder start token) |
| `PRED_LEN`  | 24 steps (~6 giờ) |

### Informer (MS mode)
- Encoder input: 10 features
- Decoder input: 10 features (label_len history + zero-padded pred_len)
- Output: 1 feature (OT only)
- Attention: **ProbSparse** (`attn='prob'`)
- Embedding: **timeF** with `freq='t'` (minutely)

---

## Anti-leakage pipeline
- STL decomposition → **fit chỉ trên Train**
- IQR Winsorization → **bounds tính từ Train**
- StandardScaler → **fit chỉ trên Train**
- Val/Test chỉ dùng `.transform()` và rolling mean (causal)
