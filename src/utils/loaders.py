import os
import pandas as pd

import zipfile
import yaml

__all__ = [
    'load_activities', 'load_locations', 'load_modalities', 'load_datasets',
    'load_data',
    'dataset_importer', 'unzip_data',
]


def load_data(fname, astype='list'):
    df = pd.read_csv(
        fname,
        delim_whitespace=True,
        header=None
    )
    if astype in {'dataframe', 'pandas', 'pd'}:
        return df
    if astype in {'values', 'np', 'numpy'}:
        return df.values
    if astype == 'list':
        return df.values.ravel().tolist()
    
    raise ValueError


def load_datasets():
    return yaml.load(open(os.path.join(
        os.environ['PROJECT_ROOT'], 'datasets.yaml'
    ), 'r'))


def load_yaml(name):
    return yaml.load(open(os.path.join(
        os.environ['PROJECT_ROOT'], f'{name}.yaml'
    ), 'r'))


def load_activities():
    return load_yaml('activities')


def load_locations():
    return load_yaml('locations')


def load_modalities():
    return load_yaml('modalities')


def dataset_importer(class_name, *args, **kwargs):
    m = __import__('src.datasets', fromlist=[class_name])
    c = getattr(m, class_name)
    return c(*args, **kwargs)


def unzip_data(zip_path, in_name, out_name):
    if os.path.exists(os.path.join(zip_path, out_name)):
        return
    with zipfile.ZipFile(os.path.join(zip_path, in_name), 'r') as fil:
        fil.extractall(zip_path)
