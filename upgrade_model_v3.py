import json

file_path = 'Report_Model.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def replace_cell(index, code):
    nb['cells'][index]['source'] = [line + '\n' for line in code.strip().split('\n')]

# Identify cells by their content
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        source = "".join(c['source'])
        
        # CELL 8: STL Decomposition (Remove it)
        if "res = stl.fit()" in source:
            replace_cell(i, "# STL Removed. We are using In-Network Decomposition now.\nprint('STL decomposition removed.')")
            
        # CELL 14: DataLoader (Fix cold-start)
        elif "class TimeSeriesDataset(Dataset):" in source and "val_ds   = TimeSeriesDataset(val_scaled," in source:
            new_code = source.replace("val_ds   = TimeSeriesDataset(val_scaled,", 
                                      "val_data_ctx = np.concatenate([train_scaled[-seq_len:], val_scaled], axis=0)\nval_ds = TimeSeriesDataset(val_data_ctx,")
            new_code = new_code.replace("test_ds  = TimeSeriesDataset(test_scaled,",
                                        "test_data_ctx = np.concatenate([val_scaled[-seq_len:], test_scaled], axis=0)\ntest_ds = TimeSeriesDataset(test_data_ctx,")
            replace_cell(i, new_code)
            
        # CELL 16: Utilities & Train Loop
        elif "def inverse_target(x, scaler, idx):" in source and "def train_model" in source:
            utils_code = """
import time
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

class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)
    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

huber_loss_fn = nn.HuberLoss(delta=1.0)
l1_loss_fn = nn.L1Loss()

def combined_loss(pred, true):
    return 0.6 * huber_loss_fn(pred, true) + 0.4 * l1_loss_fn(pred, true)

def tcn_pred_fn(model, Xb, Yb):
    f_cov = Yb[:, -pred_len:, -N_COVARIATE:]
    return model(Xb, future_features=f_cov)

def train_model(model, train_loader, val_loader, optimizer, scheduler,
                epochs, patience, model_name, save_path, pred_fn, device,
                use_noise=False, trend_lambda=0):
    train_hist, val_hist = [], []
    best_val = float('inf')
    best_epoch = 0
    counter = 0
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
            loss = combined_loss(out, y_true)
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
                vloss = combined_loss(out_v, y_val)
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
            replace_cell(i, utils_code)
            
        # CELL 19: TCN_v2
        elif "class TCN_v2(nn.Module):" in source:
            tcn_code = """
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
    def __init__(self, input_dim, num_channels, kernel_size=7,
                 dropout=0.3, horizon=24, covariate_dim=4, target_index=4):
        super().__init__()
        self.target_index = target_index
        self.revin = RevIN(num_features=1)
        self.decomp = series_decomp(25)
        
        self.trend_proj = nn.Linear(336, horizon)
        
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i - 1]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, 2**i, dropout))
        self.seasonal_network = nn.Sequential(*layers)
        last_ch = num_channels[-1]
        self.cov_proj = nn.Linear(horizon * covariate_dim, 128)
        self.seasonal_fc = nn.Sequential(
            nn.Linear(last_ch + 128, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, horizon)
        )

    def forward(self, x, future_features=None):
        target_seq = x[:, :, self.target_index:self.target_index+1]
        norm_target = self.revin(target_seq, 'norm')
        x_norm = x.clone()
        x_norm[:, :, self.target_index:self.target_index+1] = norm_target

        seasonal_init, trend_init = self.decomp(x_norm)
        
        trend_part = trend_init[:, :, self.target_index]
        trend_pred = self.trend_proj(trend_part)
        
        y = self.seasonal_network(seasonal_init.permute(0, 2, 1))
        last = y[:, :, -1]
        
        if future_features is not None:
            cov = future_features.reshape(future_features.size(0), -1)
            feat = torch.cat([last, self.cov_proj(cov)], dim=1)
        else:
            feat = torch.cat([last, torch.zeros(x.size(0), 128, device=x.device)], dim=1)
            
        seasonal_pred = self.seasonal_fc(feat)
        
        pred = trend_pred + seasonal_pred
        pred = pred.unsqueeze(-1)
        pred = self.revin(pred, 'denorm').squeeze(-1)
        return pred
            """
            replace_cell(i, tcn_code)
            
        # CELL 28: Baseline S2S
        elif "class BaselineSeq2SeqLSTM(nn.Module):" in source:
            bl_code = """
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
    def __init__(self, input_dim, hidden=128, n_layers=2, dropout=0.2,
                 dec_in_dim=None, pred_len=24, target_index=4):
        super().__init__()
        self.pred_len     = pred_len
        self.target_index = target_index
        self.revin        = RevIN(1)
        self.decomp       = series_decomp(25)
        
        self.trend_proj   = nn.Linear(336, pred_len)
        
        self.encoder      = BaselineS2SEncoder(input_dim, hidden, n_layers, dropout)
        self.cov_proj     = nn.Linear(pred_len * 4, 128)
        self.seasonal_fc  = nn.Sequential(
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

        seasonal_init, trend_init = self.decomp(x_norm)
        
        trend_part = trend_init[:, :, self.target_index]
        trend_pred = self.trend_proj(trend_part)

        enc_out, h, c = self.encoder(seasonal_init)
        last_h = h[-1]
        
        if future_cov is not None:
            cov = future_cov.reshape(future_cov.size(0), -1)
            ctx = torch.cat([last_h, self.cov_proj(cov)], dim=-1)
        else:
            ctx = torch.cat([last_h, torch.zeros(x.size(0), 128, device=x.device)], dim=-1)
            
        seasonal_pred = self.seasonal_fc(ctx)
        
        pred = trend_pred + seasonal_pred
        pred = pred.unsqueeze(-1)
        pred = self.revin(pred, 'denorm').squeeze(-1)
        return pred
            """
            replace_cell(i, bl_code)
            
        # CELL 37: Seq2SeqLSTM Attention
        elif "class Seq2SeqLSTM(nn.Module):" in source:
            s2s_code = """
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
        Q = self.q_proj(future_cov)
        K = self.k_proj(enc_out)
        V = self.v_proj(enc_out)
        
        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.bmm(attn, V)
        
        out = self.out_proj(ctx).squeeze(-1)
        return out

class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_dim, hidden=256, n_layers=2, dropout=0.2,
                 dec_in_dim=5, pred_len=24, target_index=4):
        super().__init__()
        self.pred_len     = pred_len
        self.target_index = target_index
        self.revin        = RevIN(1)
        self.decomp       = series_decomp(25)
        
        self.trend_proj   = nn.Linear(336, pred_len)
        
        self.encoder      = S2SEncoder(input_dim, hidden, n_layers, dropout)
        self.attn_decoder = DirectS2SAttention(enc_dim=hidden*2, cov_dim=4, d_model=256, pred_len=pred_len, dropout=dropout)

    def forward(self, x, y=None, future_cov=None, tf_ratio=0.0):
        target_seq = x[:, :, self.target_index:self.target_index+1]
        norm_target = self.revin(target_seq, 'norm')
        x_norm = x.clone()
        x_norm[:, :, self.target_index:self.target_index+1] = norm_target

        seasonal_init, trend_init = self.decomp(x_norm)
        
        trend_part = trend_init[:, :, self.target_index]
        trend_pred = self.trend_proj(trend_part)

        enc_out, h, c = self.encoder(seasonal_init)
        
        if future_cov is None:
            future_cov = torch.zeros(x.size(0), self.pred_len, 4, device=x.device)
            
        seasonal_pred = self.attn_decoder(enc_out, future_cov)
        
        pred = trend_pred + seasonal_pred
        pred = pred.unsqueeze(-1)
        pred = self.revin(pred, 'denorm').squeeze(-1)
        return pred
            """
            replace_cell(i, s2s_code)

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Successfully upgraded Report_Model.ipynb with Decomposition.")
