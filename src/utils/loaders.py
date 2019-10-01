import os
import pandas as pd

import yaml

__all__ = [
    # Generic
    'load_csv_data', 'load_metadata', 'build_path',
    # Metadata loaders
    'load_activities_metadata', 'load_locations_metadata',
    'load_modalities_metadata', 'load_datasets_metadata',
    'load_features_metadata', 'load_transformations_metadata',
    'load_pipelines_metadata', 'load_visualisations_metadata',
    'load_models_metadata',
    # Module importers
    'dataset_importer', 'transformer_importer', 'feature_importer',
    'pipeline_importer', 'model_importer', 'visualisation_importer',

]


# Root directory of the project


def get_root():
    """
    
    Returns:

    """
    return os.environ['PROJECT_ROOT']


# For building file structure


def build_path(*args):
    """
    
    Args:
        *args:

    Returns:

    """
    return os.path.join(
        os.environ['BUILD_ROOT'],
        *args
    )


# Generic CSV loader


def load_csv_data(fname, astype='list'):
    """
    
    Args:
        fname:
        astype:

    Returns:

    """
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


# YAML file loaders


def load_metadata(fname):
    """
    
    Args:
        fname:

    Returns:

    """
    fname = os.path.join(
        os.environ['PROJECT_ROOT'], 'metadata', fname
    )
    
    return yaml.load(open(fname, 'r'))


# Dataset metadata


def load_datasets_metadata():
    """
    
    Returns:

    """
    return load_metadata('datasets.yaml')


def load_activities_metadata():
    """
    
    Returns:

    """
    return load_metadata('activities.yaml')


def load_locations_metadata():
    """
    
    Returns:

    """
    return load_metadata('locations.yaml')


def load_modalities_metadata():
    """
    
    Returns:

    """
    return load_metadata('modalities.yaml')


# Coded metadata


def load_features_metadata():
    """
    
    Returns:

    """
    return load_metadata('features.yaml')


def load_pipelines_metadata():
    """
    
    Returns:

    """
    return load_metadata('pipelines.yaml')


def load_transformations_metadata():
    """
    
    Returns:

    """
    return load_metadata('transformers.yaml')


def load_models_metadata():
    """
    
    Returns:

    """
    return load_metadata('models.yaml')


def load_visualisations_metadata():
    """
    
    Returns:

    """
    return load_metadata('visualisations.yaml')


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
