import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader


class VeridicalDataset(Dataset):
    def __init__(self, X, labels=None, labels2id=None, id2label=None):
        self.X = X
        self.tokenizer = AutoTokenizer.from_pretrained(
            "allegro/herbert-klej-cased-tokenizer-v1"
        )
        if labels is None:
            self.labels = sorted(list(set(X['label'])))
            self.label2id = {label: i for i, label in enumerate(self.labels)}
            self.id2label = {i: label for i, label in enumerate(self.labels)}
        else:
            self.labels = labels
            self.label2id = label2id
            self.id2label = id2label
        self.n_labels = len(self.labels)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self._preprocess(X[index])
    
    def _preprocess(self, X):
        encoding = self.tokenizer(X['text'], padding='max_length', truncation=False)
        label = self.label2id(X['label'])
        return (
            torch.tensor(encoding['input_ids']),
            torch.tensor(encoding['attention_mask']),
            torch.tensor(encoding['token_type_ids']),
            torch.tensor(label)
        )