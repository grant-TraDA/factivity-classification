import logging
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from torch.utils.data import DataLoader
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

from src.data import VeridicalDataset
from src.herbert_classfier import HerBERTClassifier
from src.utils import summarize_model_per_factive


BATCH_SIZE = 32
LR = 1e-5
Y_COL = 'GOLD <T,H>'
TEXT_COL= 'T PL' # 'verb' #
N_EPOCHS = 13 # 10 #
N_SPLITS = 10 

DIR_PROJECT = Path(".").resolve()
DIR_DATA = DIR_PROJECT.joinpath("data/17_10_2021")
MODEL_DIR = DIR_PROJECT.joinpath("models")

logging.basicConfig(level=logging.INFO, filename=DIR_PROJECT.joinpath(f'log/herbert_feat_{TEXT_COL}_cv.log'))
logging.info(f"\n\nDATE: {datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}")

def transform_prediction(dataloader, model):
    _, _, embeddings = model.predict(dataloader)
    embeddings = torch.cat(embeddings, 0).cpu().numpy()
    embeddings = pd.DataFrame(embeddings, columns=['E', 'C', 'N'])
    return embeddings


df = pd.read_csv(DIR_DATA.joinpath("df.csv"))

accuracies, f1_scores, c_f1_scores, e_f1_scores, n_f1_scores = [], [], [], [], []
skf = StratifiedKFold(n_splits=N_SPLITS, random_state=123, shuffle=True)
logging.info(f"\nN splits: {N_SPLITS}, batch size: {BATCH_SIZE}, n_epochs: {N_EPOCHS}, lr: {LR}")

y, X = df[Y_COL], df.drop(Y_COL, axis=1)
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]

    MODEL = MODEL_DIR.joinpath(f"cv_herbert_{i}_{N_EPOCHS}_{LR}_{BATCH_SIZE}_{TEXT_COL}")
    model = torch.load(MODEL)

    train = [{'text': X_train.loc[i, TEXT_COL], 'label': y_train.loc[i]} for i in train_index]
    test = [{'text': X_test.loc[i, TEXT_COL], 'label': y_test.loc[i]} for i in test_index]

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

    train_embeddings = transform_prediction(train_dataloader, model)
    test_embeddings = transform_prediction(test_dataloader, model)

    X_train = X_train.drop(labels=['verb', 'T PL','H PL'], axis=1)
    X_test = X_test.drop(labels=['verb', 'T PL','H PL'], axis=1)
    
    train_df = pd.concat([df.loc[train_index, Y_COL].reset_index(drop=True), X_train.reset_index(drop=True), train_embeddings], 1)
    test_df = pd.concat([df.loc[test_index, Y_COL].reset_index(drop=True), X_test.reset_index(drop=True), test_embeddings], 1)
    # Train model
    classifier = Pipeline([
        ('model', MLPClassifier(random_state=123, learning_rate_init=LR, max_iter=4000, batch_size=BATCH_SIZE))
    ])
    classifier.fit(train_df.drop(labels=[Y_COL], axis=1), train_df[Y_COL])

    y_test_pred = classifier.predict(test_df.drop(labels=[Y_COL], axis=1))

    accuracies.append(accuracy_score(y_test, y_test_pred))
    f1_scores.append(f1_score(y_test, y_test_pred, average="weighted"))
    clas_rep = classification_report(y_test, y_test_pred, output_dict=True)
    c_f1_scores.append(clas_rep['C']["f1-score"])
    e_f1_scores.append(clas_rep['E']["f1-score"])
    n_f1_scores.append(clas_rep['N']["f1-score"])
    logging.info(f"Run: {i+1}")
    logging.info(classification_report(y_test, y_test_pred))
    summary = summarize_model_per_factive(
        y_test_pred, y_test, df.loc[X_test.index, "verb - factive/nonfactive"]
    )
    logging.info(summary)

logging.info("\nSUMMARY")
logging.info(accuracies)
logging.info(f1_scores)
logging.info(f"Accuracy: {np.mean(accuracies)} +- {np.std(accuracies)}")
logging.info(f"F1 score: {np.mean(f1_scores)} +- {np.std(f1_scores)}")
logging.info(f"F1 score C: {np.mean(c_f1_scores)} +- {np.std(c_f1_scores)}")
logging.info(f"F1 score E: {np.mean(e_f1_scores)} +- {np.std(e_f1_scores)}")
logging.info(f"F1 score N: {np.mean(n_f1_scores)} +- {np.std(n_f1_scores)}")
