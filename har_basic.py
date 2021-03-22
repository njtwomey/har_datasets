import numpy as np
from mldb import NodeWrapper
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import euclidean_distances
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.base import ExecutionGraph
from src.base import get_ancestral_metadata
from src.features.ecdf_features import ecdf
from src.features.statistical_features import statistical_features
from src.models.base import BasicScorer
from src.models.base import ClassifierWrapper
from src.models.base import instantiate_and_fit
from src.transformers.body_grav_filter import body_grav_filter
from src.transformers.resample import resample
from src.transformers.source_selector import source_selector
from src.transformers.window import window
from src.utils.loaders import dataset_importer
from src.utils.misc import randomised_order
from src.visualisations.umap_embedding import umap_embedding


def get_windowed_wearables(dataset, modality, location, fs_new, win_len, win_inc):
    selected_sources = source_selector(parent=dataset, modality=modality, location=location)
    wear_resampled = resample(parent=selected_sources, fs_new=fs_new)
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

    assert isinstance(features, NodeWrapper)

    return features


def make_classifier_node(root: ExecutionGraph, features: NodeWrapper, clf_name: str):
    if clf_name == "sgd":
        steps = [
            # ('imputation', SimpleImputer()),
            ("scaling", StandardScaler()),
            ("pca", PCA(n_components=0.95)),
            ("clf", SGDClassifier(loss="log")),
        ]

        param_grid = dict(clf__alpha=np.logspace(-5, 5, 11))

    elif clf_name == "lr":
        steps = [
            # ('imputation', SimpleImputer()),
            ("scaling", StandardScaler()),
            ("pca", PCA(n_components=0.95)),
            ("clf", LogisticRegressionCV(max_iter=1000)),
        ]

        param_grid = dict(clf__penalty=["l2"], clf__max_iter=[100])

    elif clf_name == "rf":
        steps = [
            # ('imputation', SimpleImputer()),
            ("scaling", StandardScaler()),
            ("pca", PCA(n_components=0.95)),
            ("clf", RandomForestClassifier()),
        ]

        param_grid = dict(clf__n_estimators=[10, 30, 100])

    elif clf_name == "svc":
        steps = [
            # ('imputation', SimpleImputer()),
            ("scaling", Normalizer()),
            ("pca", PCA(n_components=0.95)),
            ("minmax", MinMaxScaler(feature_range=(-1, 1))),
            ("clf", SVC(kernel="rbf", probability=True)),
        ]

        param_grid = dict(clf__gamma=["scale", "auto"], clf__C=np.logspace(-3, 3, 7),)

    else:
        raise ValueError(f"Classifier {clf_name} selected, but this is not supported.")

    key = ",".join([str(nn) for _, nn in steps]).lower()
    estimator = root.instantiate_node(key=key, func=Pipeline, kwargs=dict(steps=steps))

    return estimator, param_grid


def get_classifier(
    feature_node: NodeWrapper,
    clf_name: str,
    task_name: str,
    data_partition: str,
    train_test_split: str,
    evaluate: bool = False,
) -> ClassifierWrapper:
    # Value checks
    assert task_name in get_ancestral_metadata(feature_node, "tasks")
    assert data_partition in get_ancestral_metadata(feature_node, "data_partitions")
    assert train_test_split in get_ancestral_metadata(feature_node, "data_partitions")[data_partition]

    root: ExecutionGraph = feature_node.graph / task_name / data_partition / train_test_split

    # Instantiate the classifier
    estimator, param_grid = make_classifier_node(root=root, features=feature_node, clf_name=clf_name)

    # Instantiate the classifier
    model = ClassifierWrapper(
        parent=root,
        estimator=estimator,
        param_grid=param_grid,
        features=feature_node,
        task=feature_node.graph[task_name],
        split=root.get_split_series(data_partition=data_partition, train_test_split=train_test_split),
        scorer=BasicScorer(),
        evaluate=evaluate,
    )

    return model


def basic_har(
    #
    # Dataset
    dataset_name="anguita2013",
    #
    # Representation sources
    modality="all",
    location="all",
    #
    # Task/split
    task_name="har",
    data_partition="predefined",
    #
    # Windowification
    fs_new=33,
    win_len=3,
    win_inc=1,
    #
    # Features
    feat_name="ecdf",
    clf_name="rf",
    #
    # Embedding visualisation
    viz=False,
    evaluate=False,
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
        umap_embedding(features, task_name=task_name).evaluate()

    # Get classifier params
    models = dict()
    train_test_splits = get_ancestral_metadata(features, "data_partitions")[data_partition]
    for train_test_split in randomised_order(train_test_splits):
        models[train_test_split] = get_classifier(
            clf_name=clf_name,
            feature_node=features,
            task_name=task_name,
            data_partition=data_partition,
            evaluate=evaluate,
            train_test_split=train_test_split,
        )

    return features, models


if __name__ == "__main__":
    basic_har(viz=True, evaluate=True)
