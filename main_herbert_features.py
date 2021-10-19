import logging
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.compose import ColumnTransformer
from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from src.data import VeridicalDataset
from src.herbert_classfier import HerBERTClassifier


#MODEL = "models/herbert_13_1e-05_32_T PL"
MODEL = "models/herbert_10_1e-05_32_verb"
model = torch.load(MODEL)

BATCH_SIZE = 62
LR = 0.001
text_col= 'verb'# 'T PL' #
y_col='GOLD <T,H>'


DIR_PROJECT = Path(".").resolve()
DIR_DATA = DIR_PROJECT.joinpath("data/17_10_2021")

logging.basicConfig(level=logging.INFO, filename=Path(".").resolve().joinpath(f'log/herbert_feat_{text_col}.log'))
logging.info(f"DATE: {datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}")
logging.info(f"batch size: {BATCH_SIZE}, lr: {LR}, model: {MODEL}")

train = pd.read_csv(DIR_DATA.joinpath("train_data.csv"))
train_ = [{'text': row[text_col], 'label': row[y_col]} for _, row in train.iterrows()]
test = pd.read_csv(DIR_DATA.joinpath("test_data.csv"))
test_ = [{'text': row[text_col], 'label': row[y_col]} for _, row in test.iterrows()]
dev = pd.read_csv(DIR_DATA.joinpath("dev_data.csv"))
dev_ = [{'text': row[text_col], 'label': row[y_col]} for _, row in dev.iterrows()]


labels_ = sorted(list(set([elem['label'] for elem in train_])))
label2id_ = {label: i for i, label in enumerate(labels_)}
id2label_ = {i:label for i, label in enumerate(labels_)}

train_dataloader = DataLoader(
    VeridicalDataset(train_, labels=labels_, label2id=label2id_, id2label=id2label_),
    #VeridicalDataset(train['T PL'], labels=train['GOLD <T,H>'], labels2id=None, id2label=None),
    shuffle=True,
    batch_size = BATCH_SIZE
)
test_dataloader = DataLoader(
    VeridicalDataset(test_, labels=labels_, label2id=label2id_, id2label=id2label_),
    shuffle=False,
    batch_size = BATCH_SIZE
)
dev_dataloader = DataLoader(
    VeridicalDataset(dev_, labels=labels_, label2id=label2id_, id2label=id2label_),
    shuffle=False,
    batch_size = BATCH_SIZE
)


def transform_prediction(dataloader, model):
    _, _, embeddings = model.predict(dataloader)
    embeddings = torch.cat(embeddings, 0).cpu().numpy()
    embeddings = pd.DataFrame(embeddings, columns=['E', 'C', 'N'])
    return embeddings


train_embeddings = transform_prediction(train_dataloader, model)
test_embeddings = transform_prediction(test_dataloader, model)

# columns_model = [y_col, 'verb','verb - main semantic class','verb - tense','verb - factive/nonfactive','complement - tense','T - negation','T - type of sentence']
# train = train[columns_model]
# test = test[columns_model]
    
# cat_columns = ['verb - main semantic class', 'verb - tense', 'verb - factive/nonfactive', 'complement - tense', 'T - type of sentence']
# for colname in cat_columns:
#     if colname in train.columns:
#         train[colname] = train[colname].astype('category')
#         test[colname] = test[colname].astype('category')


X_train = train.drop(labels=['GOLD <T,H>', 'verb', 'T PL','H PL'], axis=1)
X_test = test.drop(labels=['GOLD <T,H>', 'verb', 'T PL','H PL'], axis=1)


# one_hot = ColumnTransformer([
#     ('trans', OneHotEncoder(handle_unknown="ignore"), cat_columns)
# ], sparse_threshold=0)

# train_one_hot = pd.DataFrame(one_hot.fit_transform(train), columns=one_hot.get_feature_names()) 
# test_one_hot = pd.DataFrame(one_hot.transform(test), columns=one_hot.get_feature_names())

# train = pd.concat([train['GOLD <T,H>'], train_one_hot, train_embeddings], 1)
# test = pd.concat([test['GOLD <T,H>'], test_one_hot, test_embeddings], 1)
train = pd.concat([train['GOLD <T,H>'], X_train, train_embeddings], 1)
test = pd.concat([test['GOLD <T,H>'], X_test, test_embeddings], 1)



# ### Train model
classifier = Pipeline([
    ('model', MLPClassifier(random_state=123, learning_rate_init=LR, max_iter=4000, batch_size=BATCH_SIZE))
])
classifier.fit(train.drop(labels=['GOLD <T,H>'], axis=1), train['GOLD <T,H>'])


logging.info("\nTEST")
y_pred = classifier.predict(test.drop(labels=['GOLD <T,H>'], axis=1))
logging.info(classification_report(test['GOLD <T,H>'], y_pred, digits=4))
