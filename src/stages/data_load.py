import pathlib as pl
import pandas as pd
from sklearn.datasets import load_iris

import argparse
import yaml
from typing import Text

def data_load(config_path: Text) -> None:
    with open(config_path, 'r') as conf_file:
        config = yaml.safe_load(conf_file)

    data = load_iris(as_frame=True)
    dataset = data.frame
    dataset.columns = [colname.strip(' (cm)').replace(' ', '_') for colname in dataset.columns.tolist()]
    
    dataset_csv = pl.Path(config['base']['root'], config['data']['raw_csv'])
    dataset.to_csv(dataset_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest='config', required=True, help="Path to configs file.")
    args = parser.parse_args()

    data_load(args.config)

