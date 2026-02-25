# transformer-text-classification
## Encoder-Only Transformer for Text Classification

This project implements a Transformer Encoder from scratch using PyTorch (without using `nn.Transformer` modules) to classify text. It demonstrates the internal mechanics of Multi-Head Self-Attention, Positional Encoding, and Residual Connections.

### 1. Architecture Overview

The model follows the architecture described in *Attention Is All You Need* (Vaswani et al., 2017), adapted for classification.

```text
Input Indices (Batch, Seq_Len)
      │
      ▼
Embedding Layer + Positional Encoding
      │
      ▼
┌───────────────────────────────┐
│ Transformer Encoder Block x N │
│ ┌───────────────────────────┐ │
│ │ Multi-Head Self-Attention │ │
│ │ Add & LayerNorm           │ │
│ └───────────────────────────┘ │
│ ┌───────────────────────────┐ │
│ │ Feed Forward Network      │ │
│ │ Add & LayerNorm           │ │
│ └───────────────────────────┘ │
└───────────────────────────────┘
      │
      ▼
Mean Pooling (Average over Seq_Len)
      │
      ▼
Linear Classifier (Logits)
```


v1.0 Architecture


v2.0 save pth & add interactive feature


v3.0 add evaluate reporter

v3.1 - fix: add "UNK" & "not" judgement

### 2. detail


---

#### **How to Run**

1.  **Install Requirements:**
    You only need PyTorch.
    ```bash
    pip install torch
    ```

2.  **Run Training:**
    Execute the training script. It will generate synthetic data and start training immediately.
    ```bash
    python train.py
    ```

**Expected Output:**
Since the dataset is synthetic and simple (detecting words like "good" vs "bad"), the model should converge very quickly (within 2-3 epochs) to >95% accuracy.

```text
Training on: cuda
Epoch [1/10]  Loss: 0.6950  Val Acc: 55.00%
Epoch [2/10]  Loss: 0.4501  Val Acc: 98.50%
Epoch [3/10]  Loss: 0.0523  Val Acc: 100.00%
...
```