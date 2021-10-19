import logging
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


PROJECT_PATH = Path(__file__).parent.parent.resolve()
DATA_PATH = PROJECT_PATH.joinpath("data/17_10_2021")
LOG_PATH = PROJECT_PATH.joinpath('log')
model_data_path = DATA_PATH.joinpath('ZBIOR_17.10.2021.xlsx')

logging.basicConfig(level=logging.INFO, filename=LOG_PATH.joinpath("split.log"))

model_data_path_csv = DATA_PATH.joinpath('df.csv')
train_data_path = DATA_PATH.joinpath('train_data.csv')
dev_data_path = DATA_PATH.joinpath('dev_data.csv')
test_data_path = DATA_PATH.joinpath('test_data.csv')

model_data = pd.read_excel(model_data_path, engine='openpyxl', na_values=['brak'])
model_data.dropna(how='all', axis=0, inplace=True)
model_data = model_data[model_data['GOLD <T,H>'].isin(['N', 'E', 'C'])]
model_data = model_data[model_data['verb - factive/nonfactive'].isin(['NF', 'F'])]

model_data = model_data[['T PL', 'H PL', 'verb', 'verb - main semantic class', 'verb - tense', 'verb - factive/nonfactive', 'complement - tense', 'T - negation', 'T - type of sentence', 'GOLD <T,H>']]
model_data = model_data[~(model_data['GOLD <T,H>'].isna())]

X_, X_test, y_, y_test = train_test_split(
model_data.drop(labels='GOLD <T,H>', axis=1), model_data['GOLD <T,H>'], random_state=42,
    train_size=0.8, test_size=0.2, stratify=model_data['GOLD <T,H>'])

X_train, X_dev, y_train, y_dev = train_test_split(
    X_, y_, random_state=42,
    train_size=0.875, test_size=0.125, stratify=y_
)

logging.info(f"Input data source: {model_data_path}")
logging.info(
    f"data points dimension \n{np.array((X_train.shape, X_dev.shape, X_test.shape))}"
)
logging.info(
    f"labels count\n{np.array((y_train.value_counts(), y_dev.value_counts(), y_test.value_counts()))}"
)

X_train.loc[:,'GOLD <T,H>'] = y_train
X_dev.loc[:,'GOLD <T,H>'] = y_dev
X_test.loc[:,'GOLD <T,H>'] = y_test

model_data.to_csv(model_data_path_csv, index=False)
X_train.to_csv(train_data_path, index=False)
X_dev.to_csv(dev_data_path, index=False)
X_test.to_csv(test_data_path, index=False)

logging.info(f"Data saved in directories: {train_data_path, dev_data_path, test_data_path}")
