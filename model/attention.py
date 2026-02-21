import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head

        # Linear projections for Query, Key, and Value
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Final output projection
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V):
        """
        Computes Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        """
        # Q, K, V shape: [batch_size, num_heads, seq_len, d_k]

        # 1. Dot product Q and K^T
        # K shape is [B, H, L, D], we want to transpose last two dims -> [B, H, D, L]
        # scores shape: [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 2. Apply Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # 3. Multiply by V
        # output shape: [batch_size, num_heads, seq_len, d_k]
        output = torch.matmul(attn_weights, V)

        return output, attn_weights

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 1. Linear Projections
        # Each shape: [batch_size, seq_len, d_model]
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. Split into multiple heads
        # Reshape to [batch_size, seq_len, num_heads, d_k]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k)

        # Transpose to [batch_size, num_heads, seq_len, d_k] for matrix multiplication
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 3. Calculate Attention
        attn_output, _ = self.scaled_dot_product_attention(Q, K, V)

        # 4. Concatenate heads
        # Transpose back: [batch_size, seq_len, num_heads, d_k]
        attn_output = attn_output.transpose(1, 2)

        # Flatten: [batch_size, seq_len, d_model]
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.d_model)

        # 5. Final Linear Projection
        output = self.W_o(attn_output)

        return output