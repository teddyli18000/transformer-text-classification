import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import ToyTextDataset
from model.classifier import TransformerClassifier

# ============================
# CONFIGURATION
# ============================
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 30
D_MODEL = 64  # Embedding dimension
NUM_HEADS = 2  # Number of attention heads
NUM_LAYERS = 2  # Number of encoder layers
D_FF = 256  # Feed-forward hidden dimension
SEQ_LEN = 15  # Length of sentences
NUM_SAMPLES = 5000  # Size of dataset


def train():
    # 1. Setup Data
    print("Generating synthetic dataset...")
    full_dataset = ToyTextDataset(num_samples=NUM_SAMPLES, seq_len=SEQ_LEN)
    vocab_size = full_dataset.vocab_size
    num_classes = 2  # Binary classification

    # Split Train/Val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 2. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = TransformerClassifier(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        num_classes=num_classes,
        max_len=SEQ_LEN
    ).to(device)

    # 3. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch [{epoch + 1}/{EPOCHS}]  Loss: {avg_loss:.8f}  Val Acc: {val_acc:.2f}%")
        # 在 train.py 的 train() 函数末尾添加：
        torch.save(model.state_dict(), "transformer_model.pth")
        print("模型已保存为 transformer_model.pth")


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total


if __name__ == "__main__":
    train()