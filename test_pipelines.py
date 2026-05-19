"""Test different preprocessing pipelines to find which one matches the checkpoint."""
import numpy as np, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from statsmodels.tsa.seasonal import STL
import pandas as pd

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len=336, label_len=48, pred_len=24):
        self.data, self.seq_len = data, seq_len
        self.label_len, self.pred_len = label_len, pred_len
    def __len__(self): return len(self.data) - self.seq_len - self.pred_len + 1
    def __getitem__(self, idx):
        s_end = idx + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        return torch.tensor(self.data[idx:s_end], dtype=torch.float32), torch.tensor(self.data[r_begin:r_end], dtype=torch.float32)

# Model classes
class S2SEncoder(nn.Module):
    def __init__(self, input_dim, hidden, n_layers, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0, bidirectional=True)
    def forward(self, x):
        enc_out, (h, c) = self.lstm(x)
        h = torch.cat([h[0::2], h[1::2]], dim=2)
        c = torch.cat([c[0::2], c[1::2]], dim=2)
        return enc_out, h, c

class S2SAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.scale = (hidden * 2) ** -0.5
    def forward(self, h_last, enc_out):
        q = h_last.unsqueeze(1)
        w = torch.softmax(torch.bmm(q, enc_out.transpose(1, 2)) * self.scale, dim=-1)
        return torch.bmm(w, enc_out).squeeze(1)

class S2SDecoder(nn.Module):
    def __init__(self, dec_in, hidden, n_layers, dropout=0.2):
        super().__init__()
        self.attn = S2SAttention(hidden)
        self.lstm = nn.LSTM(dec_in + hidden * 2, hidden * 2, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1))
    def forward(self, dec_in, h, c, enc_out):
        ctx = self.attn(h[-1], enc_out)
        inp = torch.cat([dec_in.squeeze(1), ctx], dim=-1).unsqueeze(1)
        out, (h, c) = self.lstm(inp, (h, c))
        return self.fc(out.squeeze(1)), h, c

class Seq2SeqModel(nn.Module):
    def __init__(self, input_dim=12, hidden=256, n_layers=2, dropout=0.2, dec_in_dim=5, pred_len=24, target_index=4):
        super().__init__()
        self.pred_len = pred_len; self.target_index = target_index
        self.encoder = S2SEncoder(input_dim, hidden, n_layers, dropout)
        self.decoder = S2SDecoder(dec_in_dim, hidden, n_layers, dropout)
    def forward(self, x, future_cov=None):
        enc_out, h, c = self.encoder(x)
        prev_out = x[:, -1, self.target_index].unsqueeze(-1)
        outputs = []
        for t in range(self.pred_len):
            cov_t = future_cov[:, t, :] if future_cov is not None else None
            dec_in = torch.cat([prev_out, cov_t], dim=-1).unsqueeze(1) if cov_t is not None else prev_out.unsqueeze(1)
            pred, h, c = self.decoder(dec_in, h, c, enc_out)
            outputs.append(pred)
            prev_out = pred
        return torch.cat(outputs, dim=1)

def add_time_features(dataframe):
    idx = dataframe.index
    t_intra = idx.hour * 4 + idx.minute // 15
    dataframe["time_sin"] = np.sin(2 * np.pi * t_intra / 96)
    dataframe["time_cos"] = np.cos(2 * np.pi * t_intra / 96)
    dataframe["day_sin"]  = np.sin(2 * np.pi * idx.dayofweek / 7)
    dataframe["day_cos"]  = np.cos(2 * np.pi * idx.dayofweek / 7)
    return dataframe

def test_pipeline(name, stl_mode, col_order):
    df = pd.read_csv("data/ETTm1.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    for col in ["MUFL", "MULL"]:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    n = len(df); train_size = int(n * 0.6); val_size = int(n * 0.2)

    if stl_mode == "full":
        stl = STL(df["OT"], period=96); res = stl.fit()
        df["trend"] = res.trend; df["seasonal"] = res.seasonal; df["residual"] = res.resid
        train_df = df.iloc[:train_size].copy()
        test_df  = df.iloc[train_size+val_size:].copy()
    else:
        train_df = df.iloc[:train_size].copy()
        test_df  = df.iloc[train_size+val_size:].copy()
        stl = STL(train_df["OT"], period=96); res = stl.fit()
        train_df["trend"] = res.trend.values
        train_df["seasonal"] = res.seasonal.values
        train_df["residual"] = res.resid.values
        pattern = np.array([res.seasonal[i::96].mean() for i in range(96)])
        test_df["trend"] = test_df["OT"].rolling(window=96, min_periods=1).mean().values
        so = (test_df.index[0].hour * 4 + test_df.index[0].minute // 15) % 96
        test_df["seasonal"] = np.array([pattern[(so + i) % 96] for i in range(len(test_df))])
        test_df["residual"] = (test_df["OT"] - test_df["trend"] - test_df["seasonal"]).values

    if col_order == "stl_first":
        # STL before time features (notebook order)
        add_time_features(train_df)
        add_time_features(test_df)
    else:
        # Time features before STL
        add_time_features(train_df)
        add_time_features(test_df)
        # Reorder: move STL to end
        cols = [c for c in train_df.columns if c not in ["trend","seasonal","residual"]] + ["trend","seasonal","residual"]
        train_df = train_df[cols]
        test_df = test_df[cols]

    target_index = list(train_df.columns).index("OT")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df.values)
    test_scaled = scaler.transform(test_df.values)

    test_loader = DataLoader(TimeSeriesDataset(test_scaled), batch_size=64, shuffle=False)

    model = Seq2SeqModel(target_index=target_index)
    model.load_state_dict(torch.load("models/best_seq2seq_v9_last_checkpoint.pth", map_location="cpu", weights_only=True))
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for Xb, Yb in test_loader:
            y_true = Yb[:, -24:, target_index]
            f_cov = Yb[:, -24:, -4:]
            out = model(Xb, future_cov=f_cov)
            preds.append(out.numpy() * scaler.scale_[target_index] + scaler.mean_[target_index])
            trues.append(y_true.numpy() * scaler.scale_[target_index] + scaler.mean_[target_index])

    all_preds = np.concatenate(preds)
    all_trues = np.concatenate(trues)
    mse = np.mean((all_preds - all_trues)**2)
    print("%-40s | Cols: %s | MSE=%.4f" % (name, list(train_df.columns), mse))
    return mse

# Test all 4 combinations
test_pipeline("A) full STL + stl_first",      "full",  "stl_first")
test_pipeline("B) full STL + time_first",      "full",  "time_first")
test_pipeline("C) train STL + stl_first",      "train", "stl_first")
test_pipeline("D) train STL + time_first",     "train", "time_first")
