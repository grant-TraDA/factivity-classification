import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

from src.data import VeridicalDataset
from src.herbert_classfier import HerBERTClassifier

BATCH_SIZE = 32
N_EPOCHS = 10
text_col='T PL'
y_col='GOLD <T,H>'


DIR_PROJECT = Path(".").resolve()
DIR_DATA = DIR_PROJECT.joinpath("data/split_dummy")

train = pd.read_csv(DIR_DATA.joinpath("train_data.csv"))
train = [{'text': row[text_col], 'label': row[y_col]} for _, row in train.iterrows()]
test = pd.read_csv(DIR_DATA.joinpath("test_data.csv"))
test = [{'text': row[text_col], 'label': row[y_col]} for _, row in test.iterrows()]
dev = pd.read_csv(DIR_DATA.joinpath("dev_data.csv"))
dev = [{'text': row[text_col], 'label': row[y_col]} for _, row in dev.iterrows()]

print(f"Train shape {len(train)}, dev {len(dev)}, test {len(test)}")


labels_ = sorted(list(set([elem['label'] for elem in train])))
label2id_ = {label: i for i, label in enumerate(labels_)}
id2label_ = {i:label for i, label in enumerate(labels_)}

train_dataloader = DataLoader(
    VeridicalDataset(train, labels=labels_, label2id=label2id_, id2label=id2label_),
    #VeridicalDataset(train['T PL'], labels=train['GOLD <T,H>'], labels2id=None, id2label=None),
    shuffle=True,
    batch_size = BATCH_SIZE
)

test_dataloader = DataLoader(
    VeridicalDataset(test, labels=labels_, label2id=label2id_, id2label=id2label_),
    shuffle=False,
    batch_size = BATCH_SIZE
)
dev_dataloader = DataLoader(
    VeridicalDataset(dev, labels=labels_, label2id=label2id_, id2label=id2label_),
    shuffle=False,
    batch_size = BATCH_SIZE
)

### Train model
classifier = HerBERTClassifier(num_labels=len(labels_))
classifier.train(train_dataloader, val_dataloader=dev_dataloader, n_epochs=N_EPOCHS, lr=1e-5)
