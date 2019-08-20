import os
import pandas as pd

import yaml

__all__ = [
    'load_activities', 'load_locations', 'load_modalities', 'load_datasets',
    'load_csv_data', 'load_features', 'load_yaml',
    'dataset_importer', 'feature_importer', 'build_path'
]


def load_csv_data(fname, astype='list'):
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


def get_root():
    return os.environ['PROJECT_ROOT']


def build_path(*args):
    return os.path.join(
        get_root(),
        *args
    )


def load_yaml(fname):
    return yaml.load(open(build_path(fname)))


def load_datasets():
    return load_yaml('datasets.yaml')


def load_activities():
    return load_yaml('activities.yaml')


def load_locations():
    return load_yaml('locations.yaml')


def load_modalities():
    return load_yaml('modalities.yaml')


def load_features():
    return load_yaml('features.yaml')


def module_importer(module_path, class_name, *args, **kwargs):
    m = __import__(module_path, fromlist=[class_name])
    c = getattr(m, class_name)
    return c(*args, **kwargs)


def dataset_importer(class_name, *args, **kwargs):
    return module_importer(
        module_path='src.datasets',
        class_name=class_name,
        *args, **kwargs
    )


def feature_importer(class_name, *args, **kwargs):
    return module_importer(
        module_path='src.features',
        class_name=class_name,
        *args, **kwargs
    )
