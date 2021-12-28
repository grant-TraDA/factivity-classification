import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from src.data import VeridicalDataset
from src.herbert_classfier import HerBERTClassifier


MODEL_DIR = Path(".").resolve().joinpath("models")
BATCH_SIZE = 32
N_EPOCHS = 13
LR = 1e-5
text_col= 'T PL' # 'verb' #
y_col='GOLD <T,H>'

DIR_PROJECT = Path(".").resolve()
DIR_DATA = DIR_PROJECT.joinpath("data/17_10_2021")

logging.basicConfig(level=logging.INFO, filename=Path(".").resolve().joinpath(f'log/herbert_{text_col}.log'))
logging.info(f"DATE: {datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}")
logging.info(f"batch size: {BATCH_SIZE}, n_epochs: {N_EPOCHS}, lr: {LR}")


train = pd.read_csv(DIR_DATA.joinpath("train_data.csv"))
train = [{'text': row[text_col], 'label': row[y_col]} for _, row in train.iterrows()]
test = pd.read_csv(DIR_DATA.joinpath("test_data.csv"))
test = [{'text': row[text_col], 'label': row[y_col]} for _, row in test.iterrows()]
dev = pd.read_csv(DIR_DATA.joinpath("dev_data.csv"))
dev = [{'text': row[text_col], 'label': row[y_col]} for _, row in dev.iterrows()]

logging.info(f"Train shape {len(train)}, dev {len(dev)}, test {len(test)}")


labels_ = sorted(list(set([elem['label'] for elem in train])))
label2id_ = {label: i for i, label in enumerate(labels_)}
id2label_ = {i:label for i, label in enumerate(labels_)}

train_dataloader = DataLoader(
    VeridicalDataset(train, labels=labels_, label2id=label2id_, id2label=id2label_),
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
classifier.train(
    train_dataloader, val_dataloader=dev_dataloader, n_epochs=N_EPOCHS, lr=LR,
    file_name=MODEL_DIR.joinpath(f"herbert_{N_EPOCHS}_{LR}_{BATCH_SIZE}_{text_col}")
)
logging.info("\nTEST")
y_pred, y_true, _ = classifier.predict(test_dataloader)
logging.info(classification_report(y_true, y_pred, digits=4))


# prediction on the entire dataset
# df_data_path = DIR_DATA.joinpath("df.csv")
# df = pd.read_csv(df_data_path)
# df_ = [{'text': row[text_col], 'label': row[y_col]} for _, row in df.iterrows()]

# df_dataloader = DataLoader(
#     VeridicalDataset(df_, labels=labels_, label2id=label2id_, id2label=id2label_),
#     shuffle=False,
#     batch_size = BATCH_SIZE
# )
# y_pred, y_true, _ = classifier.predict(df_dataloader)
# df['y_pred_herbert_sentence'] = [id2label_[elem] for elem in y_pred]

# df.to_csv(df_data_path, index=False)
