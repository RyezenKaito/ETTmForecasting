import json

file_path = 'Report_Model.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The mappings based on extracted cells_info.txt
# Cell 16: Utilities and train loops
# Cell 19: TCN_v2 model
# Cell 28: BaselineSeq2SeqLSTM model
# Cell 37: Seq2SeqLSTM model

# =====================================================================
# CELL 16 REPLACEMENT: Utilities & Train Loops (Adding RevIN, Removing trend_loss)
# =====================================================================
cell_16_code = """
def inverse_target(x, scaler, idx):
    return x * scaler.scale_[idx] + scaler.mean_[idx]

def calc_rf(k, n):
    return 1 + 2 * (k - 1) * sum(2**i for i in range(n))

def calc_metrics(y_pred, y_true):
    mse  = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    mae  = np.mean(np.abs(y_pred - y_true))
    denom = (np.abs(y_pred) + np.abs(y_true)) / 2.0
    denom = np.maximum(denom, 1.0)
    smape = np.mean(np.abs(y_pred - y_true) / denom) * 100
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'sMAPE%': smape}

class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
    def forward(self, x, mode):
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.weight + self.bias
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.bias) / self.weight
            x = x * self.stdev + self.mean
        return x

def tcn_pred_fn(model, Xb, Yb):
    f_cov = Yb[:, -pred_len:, -N_COVARIATE:]
    return model(Xb, future_features=f_cov)

def train_model(model, train_loader, val_loader, optimizer, scheduler,
                epochs, patience, model_name, save_path, pred_fn, device,
                use_noise=False, trend_lambda=0):
    criterion = nn.MSELoss()
    train_hist, val_hist = [], []
    best_val = float('inf')
    best_epoch = 0
    counter = 0
    import time
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for Xb, Yb in train_loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            y_true = Yb[:, -pred_len:, target_index]
            if use_noise and NOISE_STD > 0:
                noise = torch.randn_like(Xb) * NOISE_STD
                Xb = Xb + noise
            optimizer.zero_grad()
            out = pred_fn(model, Xb, Yb)
            loss = criterion(out, y_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xv, Yv in val_loader:
                Xv, Yv = Xv.to(device), Yv.to(device)
                y_val = Yv[:, -pred_len:, target_index]
                out_v = pred_fn(model, Xv, Yv)
                vloss = criterion(out_v, y_val)
                val_losses.append(vloss.item())
        tr = np.mean(train_losses); vl = np.mean(val_losses)
        train_hist.append(tr); val_hist.append(vl)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        ratio = vl / max(tr, 1e-9)
        print(f'[{model_name}] Ep {epoch+1:03d}/{epochs} | Train: {tr:.6f} | Val: {vl:.6f} | V/T: {ratio:.2f} | LR: {lr:.2e}')
        if vl < best_val:
            best_val = vl; best_epoch = epoch + 1
            torch.save(model.state_dict(), save_path); counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'  >> Early Stop ({model_name}) at epoch {epoch+1}'); break
    elapsed = time.time() - t0
    print(f'Best Val ({model_name}): {best_val:.6f} at epoch {best_epoch} | {elapsed:.0f}s')
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    return {'train': train_hist, 'val': val_hist, 'best_val': best_val, 'best_epoch': best_epoch, 'time': elapsed}

def train_seq2seq(model, train_loader, val_loader, optimizer, scheduler,
                  epochs, patience, save_path, device,
                  use_noise=False, trend_lambda=0):
    def s2s_pred_wrapper(m, Xb, Yb):
        f_cov = Yb[:, -pred_len:, -N_COVARIATE:]
        return m(Xb, future_cov=f_cov)
    return train_model(model, train_loader, val_loader, optimizer, scheduler,
                       epochs, patience, 'S2S', save_path, s2s_pred_wrapper, device, use_noise=use_noise)
print('Training utilities ready.')
"""

# =====================================================================
# CELL 19 REPLACEMENT: TCN_v2 (With RevIN)
# =====================================================================
cell_19_code = """
# ═══════════════════════════════════════════════════════════════
# MODEL 2: TCN_v2 (Fixed & Upgraded with RevIN)
# ═══════════════════════════════════════════════════════════════
torch.manual_seed(42); np.random.seed(42)

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

class TCN_v2(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size=5,
                 dropout=0.3, horizon=24, covariate_dim=4, target_index=4):
        super().__init__()
        self.target_index = target_index
        self.revin = RevIN(num_features=1)
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i - 1]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, 2**i, dropout))
        self.network = nn.Sequential(*layers)
        last_ch = num_channels[-1]
        self.cov_proj = nn.Linear(horizon * covariate_dim, 128)
        self.fc_head = nn.Sequential(
            nn.Linear(last_ch + 128, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, horizon)
        )

    def forward(self, x, future_features=None):
        # x: (B, T, C)
        target_seq = x[:, :, self.target_index:self.target_index+1] # (B, T, 1)
        norm_target = self.revin(target_seq, 'norm')
        x_norm = x.clone()
        x_norm[:, :, self.target_index:self.target_index+1] = norm_target

        y = self.network(x_norm.permute(0, 2, 1))
        last = y[:, :, -1]
        
        if future_features is not None:
            cov = future_features.reshape(future_features.size(0), -1)
            feat = torch.cat([last, self.cov_proj(cov)], dim=1)
        else:
            feat = torch.cat([last, torch.zeros(x.size(0), 128, device=x.device)], dim=1)
            
        pred = self.fc_head(feat)  # (B, horizon)
        pred = pred.unsqueeze(-1)  # (B, horizon, 1)
        pred = self.revin(pred, 'denorm').squeeze(-1) # (B, horizon)
        return pred
"""

# =====================================================================
# CELL 28 REPLACEMENT: BL-S2S (Direct Multi-Step + RevIN)
# =====================================================================
cell_28_code = """
# ═══════════════════════════════════════════════════════════════
# MODEL 0: BaselineSeq2SeqLSTM (Direct Multi-Step + RevIN)
# ═══════════════════════════════════════════════════════════════
torch.manual_seed(42); np.random.seed(42)

class BaselineS2SEncoder(nn.Module):
    def __init__(self, input_dim, hidden, n_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0, bidirectional=True)
    def forward(self, x):
        enc_out, (h, c) = self.lstm(x)
        h = torch.cat([h[0::2], h[1::2]], dim=2)
        c = torch.cat([c[0::2], c[1::2]], dim=2)
        return enc_out, h, c

class BaselineSeq2SeqLSTM(nn.Module):
    \"\"\"
    BiLSTM Encoder + Direct Multi-Step Projection (Baseline).
    No Attention, No Teacher Forcing. Uses RevIN.
    \"\"\"
    def __init__(self, input_dim, hidden=128, n_layers=2, dropout=0.2,
                 dec_in_dim=None, pred_len=24, target_index=4):
        super().__init__()
        self.pred_len     = pred_len
        self.target_index = target_index
        self.revin        = RevIN(1)
        self.encoder      = BaselineS2SEncoder(input_dim, hidden, n_layers, dropout)
        self.cov_proj     = nn.Linear(pred_len * 4, 128)
        self.fc           = nn.Sequential(
            nn.Linear(hidden * 2 + 128, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, pred_len)
        )

    def forward(self, x, future_cov=None, **kwargs):
        target_seq = x[:, :, self.target_index:self.target_index+1]
        norm_target = self.revin(target_seq, 'norm')
        x_norm = x.clone()
        x_norm[:, :, self.target_index:self.target_index+1] = norm_target

        enc_out, h, c = self.encoder(x_norm)
        last_h = h[-1] # (B, hidden*2)
        
        if future_cov is not None:
            cov = future_cov.reshape(future_cov.size(0), -1)
            ctx = torch.cat([last_h, self.cov_proj(cov)], dim=-1)
        else:
            ctx = torch.cat([last_h, torch.zeros(x.size(0), 128, device=x.device)], dim=-1)
            
        pred = self.fc(ctx)
        pred = pred.unsqueeze(-1)
        pred = self.revin(pred, 'denorm').squeeze(-1)
        return pred
"""

# =====================================================================
# CELL 37 REPLACEMENT: LSTM-S2S (Direct Multi-Step Attention + RevIN)
# =====================================================================
cell_37_code = """
# ═══════════════════════════════════════════════════════════════
# MODEL 1: Seq2SeqLSTM (Direct Multi-Step Attention + RevIN)
# ═══════════════════════════════════════════════════════════════
torch.manual_seed(42); np.random.seed(42)

class S2SEncoder(nn.Module):
    def __init__(self, input_dim, hidden, n_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0, bidirectional=True)
    def forward(self, x):
        enc_out, (h, c) = self.lstm(x)
        return enc_out, h, c

class DirectS2SAttention(nn.Module):
    def __init__(self, enc_dim, cov_dim, d_model, pred_len, dropout):
        super().__init__()
        self.q_proj = nn.Linear(cov_dim, d_model)
        self.k_proj = nn.Linear(enc_dim, d_model)
        self.v_proj = nn.Linear(enc_dim, d_model)
        self.scale  = d_model ** -0.5
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
    def forward(self, enc_out, future_cov):
        # enc_out: (B, T, enc_dim)
        # future_cov: (B, pred_len, cov_dim)
        Q = self.q_proj(future_cov)  # (B, pred_len, d_model)
        K = self.k_proj(enc_out)     # (B, T, d_model)
        V = self.v_proj(enc_out)     # (B, T, d_model)
        
        # Attention
        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # (B, pred_len, T)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.bmm(attn, V)  # (B, pred_len, d_model)
        
        # Predict
        out = self.out_proj(ctx).squeeze(-1)  # (B, pred_len)
        return out

class Seq2SeqLSTM(nn.Module):
    \"\"\"
    BiLSTM Encoder + Direct Multi-Step Attention Decoder.
    RevIN stabilized. No Teacher Forcing.
    \"\"\"
    def __init__(self, input_dim, hidden=256, n_layers=2, dropout=0.2,
                 dec_in_dim=5, pred_len=24, target_index=4):
        super().__init__()
        self.pred_len     = pred_len
        self.target_index = target_index
        self.revin        = RevIN(1)
        self.encoder      = S2SEncoder(input_dim, hidden, n_layers, dropout)
        self.attn_decoder = DirectS2SAttention(enc_dim=hidden*2, cov_dim=4, d_model=256, pred_len=pred_len, dropout=dropout)

    def forward(self, x, y=None, future_cov=None, tf_ratio=0.0):
        target_seq = x[:, :, self.target_index:self.target_index+1]
        norm_target = self.revin(target_seq, 'norm')
        x_norm = x.clone()
        x_norm[:, :, self.target_index:self.target_index+1] = norm_target

        enc_out, h, c = self.encoder(x_norm)
        
        if future_cov is None:
            future_cov = torch.zeros(x.size(0), self.pred_len, 4, device=x.device)
            
        pred = self.attn_decoder(enc_out, future_cov) # (B, pred_len)
        pred = pred.unsqueeze(-1)
        pred = self.revin(pred, 'denorm').squeeze(-1)
        return pred
"""

def replace_cell(index, code):
    nb['cells'][index]['source'] = [line + '\n' for line in code.strip().split('\n')]

# Assuming the indices are 16, 19, 28, 37. We will do a robust check
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        source = "".join(c['source'])
        if "def inverse_target(x, scaler, idx):" in source and "def train_model" in source:
            replace_cell(i, cell_16_code)
        elif "class TCN_v2(nn.Module):" in source:
            replace_cell(i, cell_19_code)
        elif "class BaselineSeq2SeqLSTM(nn.Module):" in source:
            replace_cell(i, cell_28_code)
        elif "class Seq2SeqLSTM(nn.Module):" in source:
            replace_cell(i, cell_37_code)

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Successfully upgraded Report_Model.ipynb")
