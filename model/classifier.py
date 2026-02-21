import torch
import torch.nn as nn
from .positional_encoding import PositionalEncoding
from .encoder import TransformerEncoder


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, num_classes, max_len=500, dropout=0.1):
        super().__init__()

        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        # 3. Transformer Encoder Stack
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, dropout)

        # 4. Classification Head
        self.classifier_head = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x shape: [batch_size, seq_len] (indices)
        """
        # Create masks if necessary (omitted for simple fixed-length toy example)

        # Embeddings -> [batch_size, seq_len, d_model]
        x = self.embedding(x)

        # Add Positional Encoding
        x = self.pos_encoder(x)

        # Pass through Transformer Encoder
        # Output: [batch_size, seq_len, d_model]
        x = self.encoder(x)

        # Pooling Strategy: Mean Pooling
        # We average over the sequence dimension to get a single vector per sentence
        x = x.mean(dim=1)

        # Classification
        x = self.dropout(x)
        logits = self.classifier_head(x)

        return logits