"""
Model definitions – extracted from Report_Model_Backup (v9 pipeline).
  1. LSTMSeq2SeqBaseline  – simple Encoder-Decoder, no attention
  2. TCN_v2               – Temporal Convolutional Network
  3. Seq2SeqLSTM          – BiLSTM + Scaled-Dot-Product Attention (best)
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# ─────────────────────────────────────────────────────────────────────────────
# 1. LSTM Seq2Seq Baseline
# ─────────────────────────────────────────────────────────────────────────────

class LSTMSeq2SeqBaseline(nn.Module):
    """
    Simplest Seq2Seq: Encoder LSTM + Decoder LSTM.
    No attention, no future covariates, basic teacher forcing.
    """
    def __init__(self, input_dim, hidden=128, n_layers=2, dropout=0.2,
                 pred_len=24, target_index=0):
        super().__init__()
        self.pred_len     = pred_len
        self.target_index = target_index

        self.encoder = nn.LSTM(input_dim, hidden, n_layers, batch_first=True,
                               dropout=dropout if n_layers > 1 else 0)
        self.decoder = nn.LSTM(1, hidden, n_layers, batch_first=True,
                               dropout=dropout if n_layers > 1 else 0)
        self.fc_out  = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_enc, y_true=None, tf_ratio=0.0, **kwargs):
        _, (h, c) = self.encoder(x_enc)
        last_val  = x_enc[:, -1, self.target_index]
        dec_input = last_val.unsqueeze(1).unsqueeze(2)

        predictions = []
        for t in range(self.pred_len):
            dec_output, (h, c) = self.decoder(dec_input, (h, c))
            pred = self.fc_out(self.dropout(dec_output[:, -1, :]))
            predictions.append(pred)
            if y_true is not None and random.random() < tf_ratio:
                dec_input = y_true[:, t].unsqueeze(1).unsqueeze(2)
            else:
                dec_input = pred.detach().unsqueeze(1)

        return torch.cat(predictions, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. TCN_v2
# ─────────────────────────────────────────────────────────────────────────────

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.norm1  = nn.LayerNorm(in_ch)
        self.norm2  = nn.LayerNorm(out_ch)
        self.conv1  = weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size,
                                            padding=pad, dilation=dilation))
        self.conv2  = weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size,
                                            padding=pad, dilation=dilation))
        self.act    = nn.GELU()
        self.drop   = nn.Dropout(dropout)
        self.proj   = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        r   = x
        out = x.transpose(1, 2); out = self.norm1(out); out = out.transpose(1, 2)
        out = self.conv1(out)[:, :, :x.size(2)]
        out = self.act(out); out = self.drop(out)
        out = out.transpose(1, 2); out = self.norm2(out); out = out.transpose(1, 2)
        out = self.conv2(out)[:, :, :x.size(2)]
        out = self.act(out); out = self.drop(out)
        res = r if self.proj is None else self.proj(r)
        return self.act(out + res)


class TCN_v2(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size=5,
                 dropout=0.3, horizon=24, covariate_dim=4, target_index=0):
        super().__init__()
        self.target_index = target_index
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i - 1]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, 2**i, dropout))
        self.network   = nn.Sequential(*layers)
        last_ch        = num_channels[-1]
        self.cov_proj  = nn.Linear(horizon * covariate_dim, 64)
        self.fc_head   = nn.Sequential(
            nn.LayerNorm(last_ch + 64),
            nn.Linear(last_ch + 64, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, horizon))

    def forward(self, x, future_features=None, **kwargs):
        y    = self.network(x.permute(0, 2, 1))
        last = y[:, :, -1]
        if future_features is not None:
            cov  = future_features.reshape(future_features.size(0), -1)
            feat = torch.cat([last, self.cov_proj(cov)], dim=1)
        else:
            feat = torch.cat([last, torch.zeros(x.size(0), 64, device=x.device)], dim=1)
        return self.fc_head(feat)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Seq2SeqLSTM – BiLSTM + Attention (best model)
# ─────────────────────────────────────────────────────────────────────────────

class S2SEncoder(nn.Module):
    def __init__(self, input_dim, hidden, n_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0,
                            bidirectional=True)

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
        w = torch.softmax(
            torch.bmm(q, enc_out.transpose(1, 2)) * self.scale, dim=-1)
        return torch.bmm(w, enc_out).squeeze(1)


class S2SDecoder(nn.Module):
    def __init__(self, dec_in, hidden, n_layers, dropout):
        super().__init__()
        self.attn = S2SAttention(hidden)
        self.lstm = nn.LSTM(dec_in + hidden * 2, hidden * 2, n_layers,
                            batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.fc   = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 1))

    def forward(self, dec_in, h, c, enc_out):
        ctx = self.attn(h[-1], enc_out)
        inp = torch.cat([dec_in.squeeze(1), ctx], dim=-1).unsqueeze(1)
        out, (h, c) = self.lstm(inp, (h, c))
        return self.fc(out.squeeze(1)), h, c


class Seq2SeqLSTM(nn.Module):
    """
    BiLSTM Encoder + Scaled-Dot-Product Attention Decoder.
    dec_in_dim = 1 (OT_prev) + N_COVARIATE (4) = 5
    Teacher forcing: probability decays 0.6 -> 0.1 over training.
    """
    def __init__(self, input_dim, hidden=256, n_layers=2, dropout=0.2,
                 dec_in_dim=5, pred_len=24, target_index=0):
        super().__init__()
        self.pred_len     = pred_len
        self.target_index = target_index
        self.encoder      = S2SEncoder(input_dim, hidden, n_layers, dropout)
        self.decoder      = S2SDecoder(dec_in_dim, hidden, n_layers, dropout)

    def forward(self, x, y=None, future_cov=None, tf_ratio=0.5, **kwargs):
        enc_out, h, c = self.encoder(x)
        prev_out      = x[:, -1, self.target_index].unsqueeze(-1)
        outputs       = []
        for t in range(self.pred_len):
            cov_t  = future_cov[:, t, :] if future_cov is not None else None
            dec_in = (torch.cat([prev_out, cov_t], dim=-1).unsqueeze(1)
                      if cov_t is not None else prev_out.unsqueeze(1))
            pred, h, c = self.decoder(dec_in, h, c, enc_out)
            outputs.append(pred)
            use_tf   = self.training and y is not None and random.random() < tf_ratio
            prev_out = y[:, t].unsqueeze(-1) if use_tf else pred
        return torch.cat(outputs, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Factory / hyperparameter configs
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "seq2seq": {
        "cls":         LSTMSeq2SeqBaseline,
        "kwargs":      dict(hidden=128, n_layers=2, dropout=0.2),
        "weight_file": "best_seq2seq_baseline.pth",
        "label":       "LSTM Seq2Seq",
    },
    "tcn": {
        "cls":         TCN_v2,
        "kwargs":      dict(num_channels=[32, 64, 128, 128, 256],
                           kernel_size=7, dropout=0.3,
                           covariate_dim=4),
        "weight_file": "best_tcn_v2.pth",
        "label":       "TCN_v2",
    },
    "attention": {
        "cls":         Seq2SeqLSTM,
        "kwargs":      dict(hidden=256, n_layers=2, dropout=0.2, dec_in_dim=5),
        "weight_file": "best_s2s_attention.pth",
        "label":       "BiLSTM + Attention",
    },
}
