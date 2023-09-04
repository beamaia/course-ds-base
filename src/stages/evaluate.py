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

def convert_to_labels(indexes, labels):
    result = []
    for i in indexes:
        result.append(labels[i])
    return result

def write_confusion_matrix_data(y_true, predicted, labels, filename):
    assert len(predicted) == len(y_true)
    predicted_labels = convert_to_labels(predicted, labels)
    true_labels = convert_to_labels(y_true, labels)
    cf = pd.DataFrame(list(zip(true_labels, predicted_labels)), columns=["y_true", "predicted"])
    cf.to_csv(filename, index=False)

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

    labels = config['data']['classes']
    cm_plot = plot_confusion_matrix(cm, labels, normalize=False)

    confusion_matrix_image = pl.Path(config['base']['root'], config['reports']['cm_img_path'])
    confusion_matrix_data_path = pl.Path(config['base']['root'], config['reports']['cm_csv_path'])

    write_confusion_matrix_data(y_test, prediction, labels=labels, filename=confusion_matrix_data_path)
    cm_plot.savefig(confusion_matrix_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest='config', required=True, help="Path to configs file.")
    args = parser.parse_args()

    evaluate(args.config)
