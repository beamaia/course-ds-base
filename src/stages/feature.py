import argparse
import yaml
import pathlib as pl
import pandas as pd

from typing import Text

def separate_features(config_path: Text) -> None:
    with open(config_path, 'r') as conf_file:
        config = yaml.safe_load(conf_file)

    dataset_path = pl.Path(config['base']['root'], config['data']['raw_csv'])
    dataset = pd.read_csv(dataset_path)

    dataset['sepal_length_to_sepal_width'] = dataset['sepal_length'] / dataset['sepal_width']
    dataset['petal_length_to_petal_width'] = dataset['petal_length'] / dataset['petal_width']

    dataset = dataset[[
        'sepal_length', 'sepal_width', 'petal_length', 'petal_width',
        'sepal_length_to_sepal_width', 'petal_length_to_petal_width',
        'target'
    ]]

    features_path = pl.Path(config['base']['root'], config['data']['features_csv'])
    dataset.to_csv(features_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest='config', required=True, help="Path to configs file.")
    args = parser.parse_args()

    separate_features(args.config)