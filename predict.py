import torch
from model.classifier import TransformerClassifier
from dataset import ToyTextDataset


def predict_sentiment(text, model, dataset, device):
    model.eval()

    # 1. 简单的 Tokenization (对应 ToyTextDataset 的词表)
    # 我们将句子拆分，并转为小写。不在词表里的词我们会忽略或用 <PAD> 代替
    words = text.lower().split()
    tokens = []
    for word in words:
        if word in dataset.vocab:
            tokens.append(dataset.vocab[word])

    # 2. 截断或填充到训练时的 SEQ_LEN (15)
    seq_len = 15
    if len(tokens) < seq_len:
        tokens += [dataset.vocab["<PAD>"]] * (seq_len - len(tokens))
    else:
        tokens = tokens[:seq_len]

    # 3. 转换为 Tensor 并移动到设备
    input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)  # Shape: [1, seq_len]

    # 4. 模型推理
    with torch.no_grad():
        logits = model(input_tensor)
        # 使用 Softmax 获取概率 (可选)
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()

    return prediction, confidence


if __name__ == "__main__":
    # --- 配置与训练时一致 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_ds = ToyTextDataset()  # 获取词表信息

    # 初始化模型结构
    model = TransformerClassifier(
        vocab_size=dummy_ds.vocab_size,
        d_model=64,
        num_heads=2,
        num_layers=2,
        d_ff=256,
        num_classes=2,
        max_len=15
    ).to(device)

    # 加载权重 (假设你在 train.py 最后保存了模型)
    # torch.save(model.state_dict(), "transformer_model.pth")
    try:
        model.load_state_dict(torch.load("transformer_model.pth", map_location=device))
        print("模型加载成功！")
    except FileNotFoundError:
        print("未找到模型文件，请先在 train.py 中添加保存代码。")
        exit()

    # --- 交互式测试 ---
    class_map = {0: "Negative (负面)", 1: "Positive (正面)"}

    while True:
        user_input = input("\n请输入一个英文句子 (输入 q 退出): ")
        if user_input.lower() == 'q':
            break

        pred, conf = predict_sentiment(user_input, model, dummy_ds, device)
        print(f"预测结果: {class_map[pred]} (置信度: {conf:.2%})")