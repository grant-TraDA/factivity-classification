import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score, f1_score
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from src.data import VeridicalDataset
from src.herbert_classfier import HerBERTClassifier
from src.utils import plot_feature_importance, summarize_model_per_factive


Y_COL = 'GOLD <T,H>'
TEXT_COL=  'verb' # 'T PL' #
N_SPLITS = 10
BATCH_SIZE = 32
N_EPOCHS = 10 # 13
LR = 1e-5

DIR_PROJECT = Path(".").resolve()
MODEL_DIR = DIR_PROJECT.joinpath("models")
DIR_DATA = DIR_PROJECT.joinpath("data/17_10_2021")

logging.basicConfig(
    level=logging.INFO, filename=DIR_PROJECT.joinpath(f'log/herbert_{TEXT_COL}_cv.log')
)

logging.info(f"\n\nDATE: {datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}")

train_ = pd.read_csv(DIR_DATA.joinpath("df.csv"))
#train = train.loc[:, [Y_COL, TEXT_COL]]

# Train model
accuracies, f1_scores, c_f1_scores, e_f1_scores, n_f1_scores = [], [], [], [], []
skf = StratifiedKFold(n_splits=N_SPLITS, random_state=123, shuffle=True)
logging.info(f"\nN splits: {N_SPLITS}, batch size: {BATCH_SIZE}, n_epochs: {N_EPOCHS}, lr: {LR}")
X, y = train_[TEXT_COL], train_[Y_COL]
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]

    train = [{'text': X_train.loc[i], 'label': y_train.loc[i]} for i in train_index]
    test = [{'text': X_test.loc[i], 'label': y_test.loc[i]} for i in test_index]

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
    
    classifier = HerBERTClassifier(num_labels=len(labels_))
    classifier.train(
        train_dataloader, n_epochs=N_EPOCHS, lr=LR,
        file_name=MODEL_DIR.joinpath(f"cv_herbert_{i}_{N_EPOCHS}_{LR}_{BATCH_SIZE}_{TEXT_COL}")
    )
    y_test_pred, y_true, _ = classifier.predict(test_dataloader)
    
    accuracies.append(accuracy_score(y_true, y_test_pred))
    f1_scores.append(f1_score(y_true, y_test_pred, average="weighted"))
    clas_rep = classification_report(y_true, y_test_pred, output_dict=True)
    c_f1_scores.append(clas_rep['0']["f1-score"])
    e_f1_scores.append(clas_rep['1']["f1-score"])
    n_f1_scores.append(clas_rep['2']["f1-score"])
    logging.info(f"Run: {i+1}")
    logging.info(classification_report(y_true, y_test_pred))
    summary = summarize_model_per_factive(
        y_test_pred, y_true, train_.loc[X_test.index, "verb - factive/nonfactive"]
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
