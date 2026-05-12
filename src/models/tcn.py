import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    Helps the model learn which feature channels are more important.
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class TemporalBlock(nn.Module):
    """
    Temporal Convolutional Block with SE Block and Residual Connection.
    (No RevIN as requested).
    """
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.norm1 = nn.LayerNorm(in_ch)
        self.norm2 = nn.LayerNorm(out_ch)
        self.conv1 = weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation))
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Add SE Block to re-weight channels
        self.se = SEBlock(out_ch)
        
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        r = x
        
        # 1st Conv Layer
        out = x.transpose(1, 2)
        out = self.norm1(out)
        out = out.transpose(1, 2)
        out = self.conv1(out)[:, :, :x.size(2)] # crop padding
        out = self.act(out)
        out = self.dropout(out)
        
        # 2nd Conv Layer
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = out.transpose(1, 2)
        out = self.conv2(out)[:, :, :x.size(2)]
        out = self.act(out)
        out = self.dropout(out)
        
        # SE Block
        out = self.se(out)
        
        # Residual Connection
        res = r if self.proj is None else self.proj(r)
        return self.act(out + res)

class TCNModel(nn.Module):
    """
    TCN Direct-Prediction Model
    - Uses 1D Dilated Convolutions
    - Future Covariates Projection
    - Deeper Fully Connected Head
    """
    def __init__(self, input_dim, num_channels, kernel_size=5,
                 dropout=0.4, horizon=24, covariate_dim=4, target_index=0):
        super().__init__()
        self.target_index = target_index
        
        # Backbone: Stack of Temporal Blocks
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i - 1]
            # dilation = 2^i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, 2**i, dropout))
        self.network = nn.Sequential(*layers)
        
        last_ch = num_channels[-1]
        
        # Covariate Projection: Map future time features to 64 dims
        self.cov_proj = nn.Linear(horizon * covariate_dim, 64)
        
        # Deeper FC Head: 2 Hidden Layers to increase capacity
        self.fc_head = nn.Sequential(
            nn.LayerNorm(last_ch + 64),
            nn.Linear(last_ch + 64, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, horizon)
        )

    def forward(self, x, future_features=None):
        """
        x: (B, T, input_dim)
        future_features: (B, horizon, covariate_dim)
        """
        # Pass through TCN backbone
        y = self.network(x.permute(0, 2, 1)) # (B, last_ch, T)
        
        # Get the feature from the last timestep
        last = y[:, :, -1] # (B, last_ch)
        
        # Incorporate future covariates if available
        if future_features is not None:
            cov = future_features.reshape(future_features.size(0), -1) # Flatten: (B, horizon * covariate_dim)
            feat = torch.cat([last, self.cov_proj(cov)], dim=1) # (B, last_ch + 64)
        else:
            feat = torch.cat([last, torch.zeros(x.size(0), 64, device=x.device)], dim=1)
            
        # Direct prediction
        pred = self.fc_head(feat)  # (B, horizon)
        return pred
