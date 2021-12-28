import joblib
import logging
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader
from sklearn.pipeline import Pipeline

from src.data import VeridicalDataset
from src.herbert_classfier import HerBERTClassifier
from src.utils import plot_feature_importance, summarize_model_per_factive


Y_COL = 'GOLD <T,H>'
N_ESTIMATORS = 100
MAX_DEPTH = 20

DIR_PROJECT = Path(".").resolve()
MODEL_DIR = DIR_PROJECT.joinpath("models")
DIR_DATA = DIR_PROJECT.joinpath("data/17_10_2021")

logging.basicConfig(level=logging.INFO, filename=DIR_PROJECT.joinpath('log/rf.log'))


train = pd.read_csv(DIR_DATA.joinpath("train_data.csv"))
test = pd.read_csv(DIR_DATA.joinpath("test_data.csv"))
dev = pd.read_csv(DIR_DATA.joinpath("dev_data.csv"))

train = train.drop(['T PL', 'H PL', 'verb'], axis=1)
test = test.drop(['T PL', 'H PL', 'verb'], axis=1)

X_train = train.drop(labels=[Y_COL], axis=1)

# Train model
classifier = RandomForestClassifier(
    n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
    class_weight={'C': 2, 'E': 1, 'N': 1}, random_state=123
)
classifier.fit(X_train, train[Y_COL])

filename = MODEL_DIR.joinpath(f'rf_n_estim_{N_ESTIMATORS}_{MAX_DEPTH}.pkl')
joblib.dump(classifier, filename)

plot_feature_importance(
    X_train.columns,
    classifier.feature_importances_,
    "feature_importance_rf.png",
    top_n=15
)

logging.info("\nTEST")
y_pred = classifier.predict(test.drop(labels=[Y_COL], axis=1))
logging.info(classification_report(test[Y_COL], y_pred, digits=4))
summary = summarize_model_per_factive(y_pred, test[Y_COL], test["verb - factive/nonfactive"])
logging.info(summary)
