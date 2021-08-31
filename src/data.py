import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer


class VeridicalDataset(Dataset):
    def __init__(
        self,
        X,
        labels=None,
        label2id=None,
        id2label=None,
        text_col='text',
        y_col='label',
    ):
        self.X = X
        self.text_col = text_col
        self.y_col = y_col
        self.tokenizer = AutoTokenizer.from_pretrained(
            "allegro/herbert-klej-cased-tokenizer-v1",
            max_len=512
        )
        if labels is None:
            labels = [elem[y_col] for elem in X]
            self.labels = sorted(list(set(labels)))
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
        return self._preprocess(self.X[index])
    
    def _preprocess(self, X):
        encoding = self.tokenizer(X[self.text_col], padding='max_length', truncation=True)
        label = self.label2id[X[self.y_col]]
        return (
            torch.tensor(encoding['input_ids']),
            torch.tensor(encoding['attention_mask']),
            torch.tensor(encoding['token_type_ids']),
            torch.tensor(label)
        )
