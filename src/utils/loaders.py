from os import environ
from pathlib import Path

import pandas as pd
import yaml
from dotenv import find_dotenv
from dotenv import load_dotenv
from loguru import logger


__all__ = [
    # Generic
    "load_csv_data",
    "load_metadata",
    "build_path",
    "get_yaml_file_list",
    "iter_dataset_paths",
    "iter_task_paths",
    "metadata_path",
    # Metadata loaders
    "load_task_metadata",
    "load_modality_metadata",
    "load_placement_metadata",
    "load_split_metadata",
    # Module importers
    "dataset_importer",
    "transformer_importer",
    "feature_importer",
    "pipeline_importer",
    "model_importer",
    "visualisation_importer",
    "load_yaml",
]


def get_env(key):
    load_dotenv(find_dotenv())
    return Path(environ[key])


# Root directory of the project
def get_project_root():
    return get_env("PROJECT_ROOT")


# For building file structure
def build_path(*args):
    path = get_env("BUILD_ROOT").joinpath(*args)
    return path


def metadata_path(*args):
    path = get_project_root() / "metadata"
    return path.joinpath(*args)


# Generic CSV loader
def load_csv_data(fname, astype="list"):
    df = pd.read_csv(fname, delim_whitespace=True, header=None)

    if astype in {"dataframe", "pandas", "pd"}:
        return df
    if astype in {"values", "np", "numpy"}:
        return df.values
    if astype == "list":
        return df.values.ravel().tolist()

    logger.exception(ValueError(f"Un-implemented type specification: {astype}"))


# YAML file loaders
def iter_files(path, suffix, stem=False):
    fil_iter = filter(lambda fil: fil.suffix == suffix, path.iterdir())
    if stem:
        yield from map(lambda fil: fil.stem, fil_iter)
    yield from map(lambda fil: path / fil, fil_iter)


def iter_dataset_paths():
    return iter_files(path=metadata_path("datasets"), suffix=".yaml", stem=False,)


def iter_task_paths():
    return iter_files(path=metadata_path("tasks"), suffix=".yaml", stem=False,)


def load_yaml(filename):
    with open(filename, "r") as fil:
        return yaml.load(fil, Loader=yaml.SafeLoader)


# Metadata
def load_metadata(*args):
    return load_yaml(metadata_path(*args))


def load_task_metadata(task_name):
    return load_metadata("task", f"{task_name}.yaml")


# Dataset metadata
def load_split_metadata():
    return load_metadata("split.yaml")


def load_placement_metadata():
    return load_metadata("placement.yaml")


def load_modality_metadata():
    return load_metadata("modality.yaml")


#


def get_yaml_file_list(*args, stem=False):
    path = metadata_path(*args)
    fil_iter = iter_files(path=path, suffix=".yaml", stem=stem)
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
    return module_importer(module_path="src.datasets", class_name=class_name, *args, **kwargs)


def feature_importer(class_name, *args, **kwargs):
    """
    
    Args:
        class_name:
        *args:
        **kwargs:

    Returns:

    """
    return module_importer(module_path="src.features", class_name=class_name, *args, **kwargs)


def transformer_importer(class_name, *args, **kwargs):
    """
    
    Args:
        class_name:
        *args:
        **kwargs:

    Returns:

    """
    return module_importer(module_path="src.transformers", class_name=class_name, *args, **kwargs)


def pipeline_importer(class_name, *args, **kwargs):
    """
    
    Args:
        class_name:
        *args:
        **kwargs:

    Returns:

    """
    return module_importer(module_path="src.pipelines", class_name=class_name, *args, **kwargs)


def model_importer(class_name, *args, **kwargs):
    """
    
    Args:
        class_name:
        *args:
        **kwargs:

    Returns:

    """
    return module_importer(module_path="src.models", class_name=class_name, *args, **kwargs)


def visualisation_importer(class_name, *args, **kwargs):
    """
    
    Args:
        class_name:
        *args:
        **kwargs:

    Returns:

    """
    return module_importer(module_path="src.visualisations", class_name=class_name, *args, **kwargs)
