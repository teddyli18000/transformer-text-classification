import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ToyTextDataset
from model.classifier import TransformerClassifier
from sklearn.metrics import classification_report, confusion_matrix


def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 准备测试数据 (增加样本量以获得准确评估)
    test_dataset = ToyTextDataset(num_samples=500, seq_len=15)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # 2. 初始化并加载训练好的模型
    model = TransformerClassifier(
        vocab_size=test_dataset.vocab_size,
        d_model=64,
        num_heads=2,
        num_layers=2,
        d_ff=256,
        num_classes=2,
        max_len=15
    ).to(device)

    try:
        model.load_state_dict(torch.load("transformer_model.pth", map_location=device))
        model.eval()
        print("Successfully loaded model weights for evaluation.\n")
    except FileNotFoundError:
        print("Error: 'transformer_model.pth' not found. Please train the model first.")
        return

    # 3. 执行评估
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    # 4. 打印学术报告级别的指标
    print("--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=["Negative", "Positive"]))

    print("--- Confusion Matrix ---")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    run_evaluation()