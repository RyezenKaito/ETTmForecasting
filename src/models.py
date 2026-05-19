"""
Model definitions – extracted from both notebooks.
  1. LSTM_Seq2Seq_Activate (from PhanPhungVu_PhamLeKhanhAn_Report35.ipynb)
  2. LSTM_Seq2Seq_Disabled (from Report_Model_Backup.ipynb)
  3. TCN_v9_Activate (from PhanPhungVu_PhamLeKhanhAn_Report35.ipynb)
  4. TCN_v9_Disabled (from Report_Model_Backup.ipynb)
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

# ─────────────────────────────────────────────────────────────────────────────
# 1. LSTM Seq2Seq
# ─────────────────────────────────────────────────────────────────────────────

class LSTM_Seq2Seq_Activate(nn.Module):
    def __init__(self, input_dim, hidden_size=128, num_layers=2,
                 pred_len=24, dropout=0.3, target_index=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_len = pred_len
        self.target_index = target_index

        self.encoder = nn.LSTM(input_dim, hidden_size, num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0)
        self.decoder = nn.LSTM(1, hidden_size, num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0)
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, x, y_teacher=None, teacher_forcing_ratio=0.5, **kwargs):
        _, (h, c) = self.encoder(x)
        decoder_input = x[:, -1, self.target_index].unsqueeze(1).unsqueeze(2)

        outputs = []
        for t in range(self.pred_len):
            dec_out, (h, c) = self.decoder(decoder_input, (h, c))
            pred_t = self.fc_out(dec_out.squeeze(1))
            outputs.append(pred_t)

            if y_teacher is not None and self.training:
                use_teacher = (torch.rand(1).item() < teacher_forcing_ratio)
                if use_teacher:
                    decoder_input = y_teacher[:, t].unsqueeze(1).unsqueeze(2)
                else:
                    decoder_input = pred_t.unsqueeze(1)
            else:
                decoder_input = pred_t.unsqueeze(1)

        return torch.cat(outputs, dim=1)


class LSTM_Seq2Seq_Disabled(nn.Module):
    def __init__(self, input_dim, hidden=128, n_layers=2, dropout=0.2,
                 pred_len=24, target_index=0):
        super().__init__()
        self.pred_len = pred_len
        self.target_index = target_index

        self.encoder = nn.LSTM(input_dim, hidden, n_layers, batch_first=True,
                               dropout=dropout if n_layers > 1 else 0)
        self.decoder = nn.LSTM(1, hidden, n_layers, batch_first=True,
                               dropout=dropout if n_layers > 1 else 0)
        self.fc_out = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_enc, y_true=None, tf_ratio=0.0, **kwargs):
        _, (h, c) = self.encoder(x_enc)
        last_val = x_enc[:, -1, self.target_index]
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
        self.norm1 = nn.LayerNorm(in_ch)
        self.norm2 = nn.LayerNorm(out_ch)
        self.conv1 = weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation))
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        r = x
        out = x.transpose(1, 2); out = self.norm1(out); out = out.transpose(1, 2)
        out = self.conv1(out)[:, :, :x.size(2)]
        out = self.act(out); out = self.dropout(out)
        out = out.transpose(1, 2); out = self.norm2(out); out = out.transpose(1, 2)
        out = self.conv2(out)[:, :, :x.size(2)]
        out = self.act(out); out = self.dropout(out)
        res = r if self.proj is None else self.proj(r)
        return self.act(out + res)


class TCN_v9_Activate(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size=5,
                 dropout=0.3, horizon=24, covariate_dim=4, target_index=0):
        super().__init__()
        self.target_index = target_index
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i - 1]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, 2**i, dropout))
        self.network = nn.Sequential(*layers)
        last_ch = num_channels[-1]
        self.cov_proj = nn.Linear(horizon * covariate_dim, 64)
        self.fc_head = nn.Sequential(
            nn.LayerNorm(last_ch + 64),
            nn.Linear(last_ch + 64, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, horizon))

    def forward(self, x, future_features=None, **kwargs):
        y = self.network(x.permute(0, 2, 1))
        last = y[:, :, -1]
        if future_features is not None:
            cov = future_features.reshape(future_features.size(0), -1)
            feat = torch.cat([last, self.cov_proj(cov)], dim=1)
        else:
            feat = torch.cat([last, torch.zeros(x.size(0), 64, device=x.device)], dim=1)
        pred = self.fc_head(feat)
        return pred


class TCN_v9_Disabled(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size=5,
                 dropout=0.3, horizon=24, covariate_dim=4, target_index=0):
        super().__init__()
        self.target_index = target_index
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i - 1]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, 2**i, dropout))
        self.network = nn.Sequential(*layers)
        last_ch = num_channels[-1]
        self.cov_proj = nn.Linear(horizon * covariate_dim, 64)
        self.fc_head = nn.Sequential(
            nn.LayerNorm(last_ch + 64),
            nn.Linear(last_ch + 64, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, horizon))

    def forward(self, x, future_features=None, **kwargs):
        y = self.network(x.permute(0, 2, 1))
        last = y[:, :, -1]
        if future_features is not None:
            cov = future_features.reshape(future_features.size(0), -1)
            feat = torch.cat([last, self.cov_proj(cov)], dim=1)
        else:
            feat = torch.cat([last, torch.zeros(x.size(0), 64, device=x.device)], dim=1)
        pred = self.fc_head(feat)
        return pred


# ─────────────────────────────────────────────────────────────────────────────
# Factory / hyperparameter configs
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "seq2seq": {
        "cls":         LSTM_Seq2Seq_Activate,
        "kwargs":      dict(hidden_size=128, num_layers=2, dropout=0.3),
        "weight_file": "G:/Code/Deep Learning/ETTForecasting/models/best_seq2seq_v9_last_checkpoint.pth",
        "label":       "LSTM Seq2Seq",
    },
    "tcn": {
        "cls":         TCN_v9_Activate,
        "kwargs":      dict(num_channels=[32, 64, 128, 128, 256],
                           kernel_size=7, dropout=0.3,
                           covariate_dim=4),
        "weight_file": "G:/Code/Deep Learning/ETTForecasting/models/best_tcn_v9_last_checkpoint.pth",
        "label":       "TCN",
    },
}
