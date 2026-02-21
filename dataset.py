import torch
from torch.utils.data import Dataset


class ToyTextDataset(Dataset):
    """
    A simple synthetic dataset for binary text classification.
    Class 0: Negative (contains words like 'bad', 'hate', 'terrible')
    Class 1: Positive (contains words like 'good', 'love', 'amazing')
    """

    def __init__(self, num_samples=1000, seq_len=10):
        self.num_samples = num_samples
        self.seq_len = seq_len

        # Simple vocabulary mapping
        self.vocab = {
            "<PAD>": 0, "I": 1, "movie": 2, "food": 3, "service": 4, "is": 5, "the": 6,
            "good": 7, "love": 8, "amazing": 9, "great": 10,  # Positive words
            "bad": 11, "hate": 12, "terrible": 13, "awful": 14  # Negative words
        }
        self.vocab_size = len(self.vocab)

        # Lists for generation
        self.pos_words = [7, 8, 9, 10]
        self.neg_words = [11, 12, 13, 14]
        self.neutral_words = [1, 2, 3, 4, 5, 6]

        self.data = []
        self.labels = []
        self._generate_data()

    def _generate_data(self):
        import random
        for _ in range(self.num_samples):
            # 50% chance of being positive
            label = random.randint(0, 1)

            # Start with some neutral filler
            sentence = [random.choice(self.neutral_words) for _ in range(self.seq_len)]

            # Insert a strong signal word based on label
            signal_word = random.choice(self.pos_words) if label == 1 else random.choice(self.neg_words)

            # Place signal word at a random position
            insert_idx = random.randint(0, self.seq_len - 1)
            sentence[insert_idx] = signal_word

            self.data.append(torch.tensor(sentence, dtype=torch.long))
            self.labels.append(torch.tensor(label, dtype=torch.long))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]