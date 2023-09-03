import pathlib as pl
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

import argparse
import yaml
from typing import Text

def train(config_path: Text) -> None:
    with open(config_path, 'r') as conf_file:
        config = yaml.safe_load(conf_file)

    random_state = config['data_split']['random_state']

    trainset_path = pl.Path(config['base']['root'], config['data_split']['trainset_path'])
    train_dataset = pd.read_csv(trainset_path)
    
    y_train = train_dataset.loc[:, 'target'].values.astype('int32')
    X_train = train_dataset.drop('target', axis=1).values.astype('float32')

    # Create an instance of Logistic Regression Classifier CV and fit the data
    clf_params = {
        'C': config['train']['C'],
        'solver': config['train']['solver'],
        'multi_class': config['train']['multi_class'],
        'max_iter': config['train']['max_iter']
    }

    logreg = LogisticRegression(**clf_params, random_state=random_state)
    logreg.fit(X_train, y_train)

    model_path= pl.Path(config['base']['root'], config['train']['model_path'])
    joblib.dump(logreg, model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest='config', required=True, help="Path to configs file.")
    args = parser.parse_args()

    train(args.config)

