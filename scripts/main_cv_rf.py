import joblib
import logging
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime

from src.data import VeridicalDataset
from src.utils import plot_feature_importance, summarize_model_per_factive


Y_COL = 'GOLD <T,H>'
N_ESTIMATORS = 100
N_SPLITS = 10
MAX_DEPTH = 20

DIR_PROJECT = Path(".").resolve()
MODEL_DIR = DIR_PROJECT.joinpath("models")
DIR_DATA = DIR_PROJECT.joinpath("data/17_10_2021")

logging.basicConfig(level=logging.INFO, filename=DIR_PROJECT.joinpath('log/rf_cv.log'))
logging.info(f"\n\nDATE: {datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}")

train = pd.read_csv(DIR_DATA.joinpath("df.csv"))
train = train.drop(['T PL', 'H PL', 'verb'], axis=1)


X_train_ = train.drop(labels=[Y_COL], axis=1)

# Train model
accuracies, f1_scores, c_f1_scores, e_f1_scores, n_f1_scores = [], [], [], [], []
skf = StratifiedKFold(n_splits=N_SPLITS, random_state=123, shuffle=True)
logging.info(f"\n\nN splits: {N_SPLITS}, N estimators: {N_ESTIMATORS}, max depth: {MAX_DEPTH}")
X, y = X_train_, train[Y_COL]
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]
    classifier = RandomForestClassifier(random_state=123, n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, class_weight={'C': 2, 'E': 1, 'N': 1}) #, class_weight={'C': 2, 'E': 1, 'N': 1})#
    classifier.fit(X_train, y_train)
    y_test_pred = classifier.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_test_pred))
    f1_scores.append(f1_score(y_test, y_test_pred, average="weighted"))
    clas_rep = classification_report(y_test, y_test_pred, output_dict=True, labels=['C', 'E', 'N'])
    c_f1_scores.append(clas_rep['C']["f1-score"])
    e_f1_scores.append(clas_rep['E']["f1-score"])
    n_f1_scores.append(clas_rep['N']["f1-score"])
    logging.info(f"Run: {i+1}")
    logging.info(classification_report(y_test, y_test_pred))
    summary = summarize_model_per_factive(
        y_test_pred, y_test, X_test.loc[:, "verb - factive/nonfactive"]
    )
    logging.info(summary)


logging.info("\nSUMMARY")
logging.info(accuracies)
logging.info(f1_scores)
logging.info(f"ACCURACY: {np.mean(accuracies)} +- {np.std(accuracies)}")
logging.info(f"F1 score: {np.mean(f1_scores)} +- {np.std(f1_scores)}")
logging.info(f"F1 score C: {np.mean(c_f1_scores)} +- {np.std(c_f1_scores)}")
logging.info(f"F1 score E: {np.mean(e_f1_scores)} +- {np.std(e_f1_scores)}")
logging.info(f"F1 score N: {np.mean(n_f1_scores)} +- {np.std(n_f1_scores)}")
