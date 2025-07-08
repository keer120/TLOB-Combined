import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(f"Input to TLOB: {x.shape}")
        # x shape: (batch_size, seq_length, d_model)
        seq_length = x.size(1)
        return x + self.pe[:, :seq_length, :]

class TLOB(nn.Module):
    def __init__(self, d_model, nhead, num_layers, seq_length, is_sin_emb=True):
        super().__init__()
        self.d_model = d_model
        transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_length) if is_sin_emb else nn.Identity()
        self.seq_length = seq_length

    def forward(self, x):
        # x shape: (batch_size, seq_length, d_model)
        if x.dim() != 3 or x.size(2) != self.d_model:
            raise ValueError(f"Expected input shape (batch_size, seq_length, d_model={self.d_model}), got {x.shape}")
        x = self.pos_encoder(x)
        x = self.transformer(x)  # Output shape: (batch_size, seq_length, d_model)
        return x