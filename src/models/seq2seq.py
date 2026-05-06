"""
src/models/seq2seq.py
Seq2SeqLSTM with Bidirectional Encoder + Scaled-Dot-Product Attention Decoder.
Extracted and cleaned up from the research notebook.
"""

import random
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Encoder  (Bidirectional LSTM)
# ─────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        """
        x : (batch, seq_len, input_dim)
        returns:
            outputs : (batch, seq_len, hidden_size * 2)
            hidden  : (num_layers, batch, hidden_size * 2)   [fwd+bwd merged]
            cell    : (num_layers, batch, hidden_size * 2)
        """
        outputs, (hidden, cell) = self.lstm(x)

        # hidden shape: (num_layers*2, batch, hidden_size)
        # Merge forward and backward by concatenating on the last dim
        hidden = torch.cat([hidden[0::2], hidden[1::2]], dim=2)
        cell   = torch.cat([cell[0::2],   cell[1::2]],   dim=2)

        return outputs, hidden, cell


# ─────────────────────────────────────────────────────────────────────────────
# Attention  (Scaled dot-product)
# ─────────────────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.scale = 1.0 / (hidden_size ** 0.5)

    def forward(self, hidden, encoder_outputs):
        """
        hidden          : (batch, hidden*2)
        encoder_outputs : (batch, seq_len, hidden*2)
        returns context : (batch, hidden*2)
        """
        q = hidden.unsqueeze(1)                                  # (batch, 1, hidden*2)
        attn = torch.bmm(q, encoder_outputs.transpose(1, 2))    # (batch, 1, seq_len)
        attn = torch.softmax(attn * self.scale, dim=-1)
        context = torch.bmm(attn, encoder_outputs)               # (batch, 1, hidden*2)
        return context.squeeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# Decoder  (Unidirectional LSTM + Attention)
# ─────────────────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.attn = Attention(hidden_size * 2)

        self.lstm = nn.LSTM(
            input_size=input_dim + hidden_size * 2,   # concat(decoder_input, context)
            hidden_size=hidden_size * 2,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, decoder_input, hidden, cell, encoder_outputs):
        """
        decoder_input   : (batch, 1, input_dim)
        returns:
            pred   : (batch, 1)
            hidden : (num_layers, batch, hidden*2)
            cell   : (num_layers, batch, hidden*2)
        """
        context = self.attn(hidden[-1], encoder_outputs)          # (batch, hidden*2)
        lstm_in = torch.cat(
            [decoder_input.squeeze(1), context], dim=1
        ).unsqueeze(1)                                            # (batch, 1, input+hidden*2)

        out, (hidden, cell) = self.lstm(lstm_in, (hidden, cell))
        pred = self.fc(out.squeeze(1))                            # (batch, 1)
        return pred, hidden, cell


# ─────────────────────────────────────────────────────────────────────────────
# Seq2SeqLSTM
# ─────────────────────────────────────────────────────────────────────────────

class Seq2SeqLSTM(nn.Module):
    """
    Encoder–Decoder LSTM with:
      - Bidirectional encoder
      - Scaled dot-product attention in decoder
      - Teacher forcing (scheduled sampling with decay)
      - Correct feature-order reconstruction at each decoder step

    Args:
        input_dim  : number of input features (N_FEATURES from config)
        hidden_size: LSTM hidden units per direction
        num_layers : stacked LSTM layers
        pred_len   : forecast horizon
        dropout    : dropout probability
        target_idx : column index of the target variable (OT) in the feature array
    """

    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float, dec_in_dim: int, pred_len: int, target_idx: int):
        super().__init__()
        self.encoder    = Encoder(input_dim, hidden_size, num_layers, dropout)
        self.decoder    = Decoder(dec_in_dim, hidden_size, num_layers, dropout)
        self.pred_len   = pred_len
        self.target_idx = target_idx

    def forward(self, x, y=None, future_features=None, teacher_forcing_ratio: float = 0.5):
        """
        x                     : (batch, seq_len, input_dim)   — encoder input
        y                     : (batch, pred_len)              — ground-truth targets
                                  (used for teacher forcing, can be None at inference)
        future_features       : (batch, pred_len, input_dim-1) — known future covariates
                                  (time_sin/cos, etc.)
        teacher_forcing_ratio : probability of using ground truth at each step
        returns outputs       : (batch, pred_len)
        """
        # ── Encoder ──────────────────────────────────────────────────────────
        encoder_outputs, hidden, cell = self.encoder(x)

        # ── Initial decoder input: last timestep of the encoder input ────────
        # We must construct a 3-feature input: [target, time_sin, time_cos] to match dec_in_dim=3
        dec_init_target = x[:, -1:, self.target_idx:self.target_idx+1]
        dec_init_covariates = x[:, -1:, -2:]   # time_sin, time_cos (always last 2 cols)
        decoder_input = torch.cat([dec_init_target, dec_init_covariates], dim=-1)   # (batch, 1, 3)

        outputs = []

        # ── Decoder loop ─────────────────────────────────────────────────────
        for t in range(self.pred_len):
            pred, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            outputs.append(pred.unsqueeze(1))   # (batch, 1, 1)

            # ── Scheduled sampling ────────────────────────────────────────
            if self.training and y is not None and random.random() < teacher_forcing_ratio:
                next_target = y[:, t].unsqueeze(1).unsqueeze(2)   # (batch, 1, 1) ground truth
            else:
                next_target = pred.unsqueeze(1)                    # (batch, 1, 1) model pred

            # ── Build next decoder input with correct feature ordering ────
            if future_features is not None:
                # future_features should now be shape (batch, pred_len, 2) containing only sin and cos
                covariates = future_features[:, t:t+1, :]          # (batch, 1, 2)
            else:
                # Fallback: reuse sin, cos from last encoder step (index 7 and 8 in the 9-feature array)
                covariates = x[:, -1:, -2:]   # fallback: sin, cos from last encoder step

            # decoder_input will have shape (batch, 1, 3): [next_target, sin, cos]
            decoder_input = torch.cat([
                next_target,
                covariates
            ], dim=-1)   # (batch, 1, 3)

        outputs = torch.cat(outputs, dim=1)   # (batch, pred_len, 1)
        return outputs.squeeze(-1)            # (batch, pred_len)
