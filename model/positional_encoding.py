import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Injects information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension d_model
    as the embeddings, so that the two can be summed.
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Create a matrix of [max_len, d_model] representing positional encodings
        pe = torch.zeros(max_len, d_model)

        # position: [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # div_term: calculate 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension: [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # Register as buffer (it's not a learnable parameter, but part of state_dict)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        Returns:
            x + positional_encoding
        """
        # Add positional encoding to the input embedding
        # We slice self.pe to match the sequence length of x
        x = x + self.pe[:, :x.size(1), :]
        return x