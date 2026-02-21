import torch
import torch.nn as nn
from .attention import MultiHeadAttention


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # Simple two-layer MLP with ReLU activation
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Sub-layer 1: Multi-Head Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Sub-layer 2: Feed Forward
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Standard Post-LN Transformer Layer implementation
        x = LayerNorm(x + Sublayer(x))
        """
        # 1. Multi-Head Attention part
        # Save residual
        residual = x
        x = self.self_attn(x)
        x = self.dropout1(x)
        # Add & Norm
        x = self.norm1(x + residual)

        # 2. Feed Forward part
        residual = x
        x = self.feed_forward(x)
        x = self.dropout2(x)
        # Add & Norm
        x = self.norm2(x + residual)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x