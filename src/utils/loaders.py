from os.path import join
from os import environ, listdir

import pandas as pd

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    # Generic
    'load_csv_data', 'load_metadata', 'build_path', 'get_yaml_file_list',
    'iter_datasets', 'iter_tasks',
    # Metadata loaders
    'load_task_metadata', 'load_modality_metadata', 'load_placement_metadata',
    'load_split_metadata',
    # Module importers
    'dataset_importer', 'transformer_importer', 'feature_importer',
    'pipeline_importer', 'model_importer', 'visualisation_importer',

]


# Root directory of the project

def get_project_root():
    return environ['PROJECT_ROOT']


# For building file structure

def build_path(*args):
    return join(
        environ['BUILD_ROOT'],
        *args
    )


def metadata_path():
    return join(
        get_project_root(),
        'metadata'
    )


# Generic CSV loader

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
    
    logger.exception(ValueError(
        f"Un-implemented type specification: {astype}"
    ))


# YAML file loaders
def iter_files(path, ext, strip_ext=False):
    fil_iter = filter(lambda fil: fil.endswith(ext), listdir(path))
    if strip_ext:
        return map(lambda fil: fil[:-len(ext)], fil_iter)
    return fil_iter


def iter_datasets():
    return iter_files(
        path=join(metadata_path(), 'datasets'),
        ext='.yaml', strip_ext=True
    )


def iter_tasks():
    return iter_files(
        path=join(metadata_path(), 'tasks'),
        ext='.yaml', strip_ext=True
    )


# Metadata
def load_metadata(fname):
    return yaml.load(open(join(metadata_path(), fname), 'r'))


def load_task_metadata(task_name):
    return load_metadata(join('tasks', f'{task_name}.yaml'))


# Dataset metadata
def load_split_metadata():
    return load_metadata('split.yaml')


def load_placement_metadata():
    return load_metadata('placement.yaml')


def load_modality_metadata():
    return load_metadata('modality.yaml')


#

def get_yaml_file_list(*args, strip_ext=False):
    path = join(metadata_path(), *args)
    fil_iter = iter_files(path=path, ext='.yaml', strip_ext=strip_ext)
    return list(fil_iter)


# Module importers


def module_importer(module_path, class_name, *args, **kwargs):
    """
    
    Args:
        module_path:
        class_name:
        *args:
        **kwargs:

    Returns:

    """
    m = __import__(module_path, fromlist=[class_name])
    c = getattr(m, class_name)
    return c(*args, **kwargs)


def dataset_importer(class_name, *args, **kwargs):
    """
    
    Args:
        class_name:
        *args:
        **kwargs:

    Returns:

    """
    return module_importer(
        module_path='src.datasets',
        class_name=class_name,
        *args, **kwargs
    )


def feature_importer(class_name, *args, **kwargs):
    """
    
    Args:
        class_name:
        *args:
        **kwargs:

    Returns:

    """
    return module_importer(
        module_path='src.features',
        class_name=class_name,
        *args, **kwargs
    )


def transformer_importer(class_name, *args, **kwargs):
    """
    
    Args:
        class_name:
        *args:
        **kwargs:

    Returns:

    """
    return module_importer(
        module_path='src.transformers',
        class_name=class_name,
        *args, **kwargs
    )


def pipeline_importer(class_name, *args, **kwargs):
    """
    
    Args:
        class_name:
        *args:
        **kwargs:

    Returns:

    """
    return module_importer(
        module_path='src.pipelines',
        class_name=class_name,
        *args, **kwargs
    )


def model_importer(class_name, *args, **kwargs):
    """
    
    Args:
        class_name:
        *args:
        **kwargs:

    Returns:

    """
    return module_importer(
        module_path='src.models',
        class_name=class_name,
        *args, **kwargs
    )


def visualisation_importer(class_name, *args, **kwargs):
    """
    
    Args:
        class_name:
        *args:
        **kwargs:

    Returns:

    """
    return module_importer(
        module_path='src.visualisations',
        class_name=class_name,
        *args, **kwargs
    )
