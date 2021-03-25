import numpy as np
from mldb import NodeWrapper
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.base import ExecutionGraph
from src.base import get_ancestral_metadata
from src.models.base import BasicScorer
from src.models.base import ClassifierWrapper


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
