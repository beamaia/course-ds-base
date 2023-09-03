import pathlib as pl
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import joblib

import argparse
import yaml
import json
from typing import Text

from src.report.visualize import plot_confusion_matrix

def evaluate(config_path: Text) -> None:
    with open(config_path, 'r') as conf_file:
        config = yaml.safe_load(conf_file)

    testset_path = pl.Path(config['base']['root'], config['data_split']['testset_path'])
    test_dataset = pd.read_csv(testset_path)
    
    y_test = test_dataset.loc[:, 'target'].values.astype('int32')
    X_test = test_dataset.drop('target', axis=1).values.astype('float32')

    model_path= pl.Path(config['base']['root'], config['train']['model_path'])
    logreg = joblib.load(model_path)

    prediction = logreg.predict(X_test)
    cm = confusion_matrix(prediction, y_test)
    f1 = f1_score(y_true = y_test, y_pred = prediction, average='macro')

    metrics_file = pl.Path(config['base']['root'], config['reports']['metrics_path'])

    metrics = {
        'f1': f1
    }

    with open(metrics_file, 'w') as mf:
        json.dump(
            obj=metrics,
            fp=mf,
            indent=4
        )

    cm_plot = plot_confusion_matrix(cm, config['data']['classes'], normalize=False)
    confusion_matrix_image = pl.Path(config['base']['root'], config['reports']['cm_path'])
    cm_plot.savefig(confusion_matrix_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest='config', required=True, help="Path to configs file.")
    args = parser.parse_args()

    evaluate(args.config)
