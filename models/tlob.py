import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TLOB(nn.Module):
    def __init__(self, hidden_dim, num_layers, seq_size, num_features, num_heads, is_sin_emb, dataset_type, num_classes=3):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=num_features,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(num_features, num_classes)
        self.pos_encoder = PositionalEncoding(num_features, seq_size) if is_sin_emb else None

    def forward(self, x):
        if self.pos_encoder:
            x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # Take last time step
        x = self.output_layer(x)
        return x