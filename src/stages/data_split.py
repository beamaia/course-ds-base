import pathlib as pl
import pandas as pd
from sklearn.model_selection import train_test_split

import argparse
import yaml
from typing import Text

def split_dataset(config_path: Text) -> None:
    with open(config_path, 'r') as conf_file:
        config = yaml.safe_load(conf_file)

    dataset_path = pl.Path(config['base']['root'], config['data']['features_csv'])
    dataset = pd.read_csv(dataset_path)

    random_state = config['data_split']['random_state']
    test_size = config['data_split']['test_size']

    train_dataset, test_dataset = train_test_split(dataset, test_size=test_size, random_state=random_state)
    train_dataset.shape, test_dataset.shape

    # Save train and test sets
    trainset_path = pl.Path(config['base']['root'], config['data_split']['trainset_path'])
    testset_path = pl.Path(config['base']['root'], config['data_split']['testset_path'])

    train_dataset.to_csv(trainset_path)
    test_dataset.to_csv(testset_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest='config', required=True, help="Path to configs file.")
    args = parser.parse_args()

    split_dataset(args.config)

