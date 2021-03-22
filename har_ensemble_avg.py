import numpy as np
from mldb import NodeWrapper
from sklearn.base import BaseEstimator
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Bunch

from har_basic import get_classifier
from har_basic import get_features
from har_basic import get_windowed_wearables
from src.base import ExecutionGraph
from src.base import get_ancestral_metadata
from src.models.base import BasicScorer
from src.models.base import ClassifierWrapper
from src.utils.loaders import dataset_importer
from src.utils.misc import randomised_order


class PrefittedVotingClassifier(BaseEstimator):
    def __init__(self, estimators, voting="soft", weights=None, verbose=False):
        assert weights is None or len(weights) == len(estimators)
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.verbose = verbose
        self.le_ = None

    def predict_proba(self, X):
        weights = self.weights
        if weights is None:
            weights = np.ones(len(weights))
        probs = [est.predict_proba(X) * ww for ww, (_, est) in zip(weights, self.estimators)]
        return sum(probs)

    def predict(self, X):
        probs = self.predict_proba(X)
        inds = np.argmax(probs, axis=1)
        return self.le_.classes_[inds]

    def fit(self, X, y, sample_weight=None):
        self.le = LabelEncoder().fit(y)
        return self

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


def ensemble_classifier(
    task_name: str,
    feat_name: str,
    data_partition: str,
    train_test_split: str,
    windowed_data: NodeWrapper,
    clf_names,
    evaluate=False,
):
    features = get_features(feat_name=feat_name, windowed_data=windowed_data)

    graph: ExecutionGraph = features.graph / f"ensemble-over={sorted(clf_names)}" / task_name / train_test_split

    estimators = list()
    for clf_name in randomised_order(clf_names):
        estimators.append(
            [
                f"{clf_name=}",
                get_classifier(
                    feature_node=features,
                    clf_name=clf_name,
                    task_name=task_name,
                    data_partition=data_partition,
                    train_test_split=train_test_split,
                ).model,
            ]
        )

    estimator = graph.instantiate_node(
        key=f"PrefittedVotingClassifier-{feat_name}".lower(),
        func=PrefittedVotingClassifier,
        kwargs=dict(estimators=estimators, voting="soft", verbose=10,),
    )

    param_grid = None
    # param_grid = dict(weights=list(map(tuple, np.random.dirichlet(np.ones(len(estimators)), size=20))))

    model = ClassifierWrapper(
        parent=graph,
        estimator=estimator,
        param_grid=param_grid,
        features=features,
        task=features.graph[task_name],
        split=graph.get_split_series(data_partition=data_partition, train_test_split=train_test_split),
        scorer=BasicScorer(),
        evaluate=evaluate,
    )

    return model


def basic_ensemble(
    dataset_name="anguita2013",
    modality="all",
    location="all",
    task_name="har",
    feat_name="ecdf",
    data_partition="predefined",
    fs_new=33,
    win_len=3,
    win_inc=1,
):
    dataset = dataset_importer(dataset_name)

    windowed_data = get_windowed_wearables(
        dataset=dataset, modality=modality, location=location, fs_new=fs_new, win_len=win_len, win_inc=win_inc
    )

    models = dict()

    train_test_splits = get_ancestral_metadata(windowed_data, "data_partitions")[data_partition]
    for train_test_split in randomised_order(train_test_splits):
        models[train_test_split] = ensemble_classifier(
            feat_name=feat_name,
            task_name=task_name,
            data_partition=data_partition,
            windowed_data=windowed_data,
            train_test_split=train_test_split,
            clf_names=["sgd", "lr", "rf"],
            evaluate=True,
        )

    return models


if __name__ == "__main__":
    basic_ensemble()
