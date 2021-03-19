import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler

from src.features import ecdf
from src.features import statistical_features
from src.models.base import instantiate_classifiers
from src.transformers import body_grav_filter
from src.transformers import resample
from src.transformers import window
from src.transformers.source_selector import concatenate_features
from src.transformers.source_selector import source_selector
from src.utils.loaders import dataset_importer
from src.visualisations import umap_embedding


def get_windowed_wearables(dataset, modality, location, fs_new, win_len, win_inc):
    selectetd_sources = source_selector(parent=dataset, modality=modality, location=location)
    wear_resampled = resample(parent=selectetd_sources, fs_new=fs_new)
    wear_filtered = body_grav_filter(parent=wear_resampled)
    wear_windowed = window(parent=wear_filtered, win_len=win_len, win_inc=win_inc)
    return wear_windowed


def get_features(feat_name, windowed_data):
    if feat_name == "statistical":
        features = statistical_features(parent=windowed_data)
    elif feat_name == "ecdf":
        features = ecdf(parent=windowed_data, n_components=21)
    else:
        raise ValueError
    return concatenate_features(features)


def get_classifier(clf_name, features, task_name, split_name):
    if clf_name == "sgd":
        estimator = Pipeline([("scaling", Normalizer()), ("clf", SGDClassifier(loss="log"))])
        param_grid = dict(clf__alpha=np.logspace(-5, 5, 11))
    elif clf_name == "lr":
        estimator = Pipeline(
            [["scaling", StandardScaler()], ("pca", PCA(0.9)), ("clf", LogisticRegressionCV(max_iter=100))]
        )
        param_grid = None
    elif clf_name == "rf":
        estimator = Pipeline([("clf", RandomForestClassifier())])
        param_grid = dict(clf__n_estimators=[10, 30, 100])
    else:
        raise ValueError

    task, target, splits, model_nodes = instantiate_classifiers(
        index=features.graph["index"],
        features=features,
        task_name=task_name,
        split_name=split_name,
        estimator=estimator,
        param_grid=param_grid,
    )

    return task, target, splits, model_nodes


def basic_har(
    #
    # Dataset
    dataset_name="pamap2",
    #
    # Representation sources
    modality="all",
    location="all",
    #
    # Task/split
    task_name="har",
    split_name="predefined",
    #
    # Windowification
    fs_new=33,
    win_len=3,
    win_inc=1,
    #
    # Features
    feat_name="statistical",
    clf_name="lr",
    #
    # Embedding visualisation
    viz=True,
):
    dataset = dataset_importer(dataset_name)

    # Resample, filter and window the raw sensor data
    wear_windowed = get_windowed_wearables(
        dataset=dataset, modality=modality, location=location, fs_new=fs_new, win_len=win_len, win_inc=win_inc
    )

    # Extract features
    features = get_features(feat_name=feat_name, windowed_data=wear_windowed)

    # Visualise the feature embeddings
    if viz:
        umap_embedding(features, task_name=task_name)

    # Get classifier params
    task, target, splits, model_nodes = get_classifier(
        clf_name=clf_name, features=features, task_name=task_name, split_name=split_name
    )

    return features, task_name, target, splits, model_nodes


if __name__ == "__main__":
    basic_har()
