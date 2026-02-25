import torch
from torch.utils.data import Dataset


class ToyTextDataset(Dataset):
    """
    A simple synthetic dataset for binary text classification.
    Class 0: Negative (contains words like 'bad', 'hate', 'terrible')
    Class 1: Positive (contains words like 'good', 'love', 'amazing')
    """

    def __init__(self, num_samples=1000, seq_len=15):
        self.num_samples = num_samples
        self.seq_len = seq_len

        # Simple vocabulary mapping
        self.vocab = {
            "<PAD>": 0,"<UNK>":1, "i": 2, "movie": 3, "food": 4, "service": 5, "is": 6, "the": 7,
            "good": 8, "love": 9, "amazing": 10, "great": 11,"like":12,  # Positive words
            "bad": 13, "hate": 14, "terrible": 15, "awful": 16,"not":17  # Negative words
        }
        self.vocab_size = len(self.vocab)

        # Lists for generation
        self.pos_words = [8, 9, 10,11,12]
        self.neg_words = [13, 14,15,16]
        self.neutral_words = [2, 3, 4, 5, 6,7]
        self.not_idx=17

        self.data = []
        self.labels = []
        self._generate_data()

    def _generate_data(self):
        import random
        for _ in range(self.num_samples):
            # 50% chance of being positive
            label = random.randint(0, 1)
            # 生成基础中性句子
            sentence = [random.choice(self.neutral_words) for _ in range(self.seq_len)]

            if label == 1:
                # 正向：插入一个正向词，且确保它前面没有 'not'
                signal_word = random.choice(self.pos_words)
                insert_idx = random.randint(1, self.seq_len - 1)
                sentence[insert_idx] = signal_word
            else:
                # 负向：有两种生成方式
                if random.random() < 0.5:
                    # 方式 A: 直接插入传统的负面词 (bad, hate...)
                    signal_word = random.choice(self.neg_words)
                    insert_idx = random.randint(0, self.seq_len - 1)
                    sentence[insert_idx] = signal_word
                else:
                    # 方式 B: 逻辑反转！插入 "not" + "positive_word" (例如 "not like")
                    pos_word = random.choice(self.pos_words)
                    insert_idx = random.randint(1, self.seq_len - 1)
                    sentence[insert_idx] = pos_word
                    sentence[insert_idx - 1] = self.not_idx  # 在正向词前放个 'not'

            self.data.append(torch.tensor(sentence, dtype=torch.long))
            self.labels.append(torch.tensor(label, dtype=torch.long))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]