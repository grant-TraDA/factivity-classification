import joblib
import logging
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from src.data import VeridicalDataset
from src.herbert_classfier import HerBERTClassifier
from src.utils import plot_feature_importance



Y_COL = 'GOLD <T,H>'
N_ESTIMATORS = 100

DIR_PROJECT = Path(".").resolve()
MODEL_DIR = DIR_PROJECT.joinpath("models")
DIR_DATA = DIR_PROJECT.joinpath("data/17_10_2021")

logging.basicConfig(level=logging.INFO, filename=DIR_PROJECT.joinpath('log/rf.log'))


train = pd.read_csv(DIR_DATA.joinpath("train_data.csv"))
test = pd.read_csv(DIR_DATA.joinpath("test_data.csv"))
dev = pd.read_csv(DIR_DATA.joinpath("dev_data.csv"))

train = train.drop(['T PL', 'H PL', 'verb'], axis=1)
test = test.drop(['T PL', 'H PL', 'verb'], axis=1)
  

#columns_model = [Y_COL, 'verb','verb - main semantic class','verb - tense','verb - factive/nonfactive','complement - tense','T - negation','T - type of sentence']

# train = train[columns_model]
# test = test[columns_model]
    
# for colname in ['verb', 'verb - main semantic class', 'verb - tense', 'verb - factive/nonfactive', 'complement - tense', 'T - type of sentence']:
#     if colname in train.columns:
#         train[colname] = train[colname].astype('category')
#         test[colname] = test[colname].astype('category')

#columns_to_dummify = [column for column in train.columns if ((column != Y_COL) and (column != "verb"))]

X_train = train.drop(labels=[Y_COL], axis=1)

#from sklearn.preprocessing import OrdinalEncoder
# Train model
classifier = RandomForestClassifier(random_state=10, n_estimators=N_ESTIMATORS)
# classifier = Pipeline([
# #    ('trans', OneHotEncoder(handle_unknown="ignore")),
#     #('trans', OrdinalEncoder(handle_unknown= "use_encoded_value", unknown_value=-1)),
#     ('model', RandomForestClassifier(random_state=10, n_estimators=N_ESTIMATORS))
# ])
classifier.fit(X_train, train[Y_COL])

filename = MODEL_DIR.joinpath(f'rf_n_estim_{N_ESTIMATORS}.pkl')
joblib.dump(classifier, filename)

#feat = [item for sublist in classifier.named_steps['trans'].categories_ for item in sublist]
plot_feature_importance(
    X_train.columns,
    classifier.feature_importances_,
    "feature_importance_rf.png",
    top_n=15
)

logging.info("\nTEST")
y_pred = classifier.predict(test.drop(labels=[Y_COL], axis=1))
logging.info(classification_report(test[Y_COL], y_pred, digits=4))

# prediction on the entire dataset
# df_data_path = DIR_DATA.joinpath("df.csv")
# df = pd.read_csv(df_data_path)
# y_pred = classifier.predict(df[list(X_train.columns)])
# df['y_pred_rf'] = y_pred

#df.to_csv(df_data_path, index=False)
