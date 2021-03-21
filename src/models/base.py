from typing import Any
from typing import Dict

import numpy as np
from loguru import logger
from mldb import NodeWrapper
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline

from src.base import ExecutionGraph
from src.base import get_ancestral_metadata
from src.evaluation.classification import evaluate_data_split
from src.utils.misc import randomised_order

__all__ = ["instantiate_and_fit", "ClassifierWrapper", "instantiate_classifiers"]


def get_estimator_name(estimator):
    if isinstance(estimator, Pipeline):
        parts = []
        for key, sub_estimator in estimator.steps:
            sub_estimator_name = get_estimator_name(sub_estimator)
            parts.append(sub_estimator_name)
        internal = ", ".join(parts)
        return f"Pipeline({internal})"

    elif isinstance(estimator, FeatureUnion):
        raise NotImplementedError

    name = str(estimator)

    # TODO: Remove newlines, and extra spaces

    return name


def instantiate_and_fit(index, fold, X, y, estimator, n_splits=5, param_grid=None):
    assert fold.shape[0] == index.shape[0]
    assert fold.shape[0] == X.shape[0]
    assert fold.shape[0] == y.shape[0]

    fold_vals = fold.ravel()

    train_inds = fold_vals == "train"
    val_inds = fold_vals == "val"

    if val_inds.sum():
        raise NotImplementedError("Explicit validation indices not yet supported.")

    y = y.values.ravel()

    nan_row, nan_col = np.nonzero(np.isnan(X) | np.isinf(X))
    if len(nan_row):
        logger.warning(f"Setting {len(nan_row)} NaN elements to zero before fitting {estimator}.")
        X[nan_row, nan_col] = 0

    estimator_clone = clone(estimator)

    logger.info(f"Fitting {estimator_clone} on data (shape: {X.shape})")

    if param_grid is not None:
        grid_search = GridSearchCV(estimator=estimator_clone, param_grid=param_grid, verbose=10)
        k_fold = GroupKFold(n_splits=n_splits).split(X[train_inds], y[train_inds], index.trial.values[train_inds])
        grid_search.cv = list(k_fold)
        return grid_search.fit(X[train_inds], y[train_inds])

    return estimator_clone.fit(X[train_inds], y[train_inds])


# noinspection PyPep8Naming
class ClassifierWrapper(ExecutionGraph):
    def __init__(
        self,
        parent: ExecutionGraph,
        features: NodeWrapper,
        split: NodeWrapper,
        task: NodeWrapper,
        estimator: Any,
        param_grid: Any,
    ):
        assert isinstance(parent, ExecutionGraph)
        assert isinstance(features, NodeWrapper)
        assert isinstance(split, NodeWrapper)
        assert isinstance(task, NodeWrapper)

        super(ClassifierWrapper, self).__init__(parent=parent, name=get_estimator_name(estimator))

        self.estimator = estimator
        self.param_grid = param_grid

        self.features = features
        self.split = split
        self.task = task

        model = self.instantiate_node(
            key="model",
            func=instantiate_and_fit,
            backend="sklearn",
            kwargs=dict(
                X=features,
                y=task,
                index=self["index"],
                fold=self.split,
                estimator=self.estimator,
                param_grid=self.param_grid,
            ),
        )

        self.get_or_create(
            key="results",
            func=evaluate_data_split,
            backend="json",
            kwargs=dict(split=split, targets=task, estimator=model, prob_predictions=self.predict_proba(features)),
        )

    @property
    def model(self):
        return self["model"]

    @property
    def results(self):
        return self["results"]

    def score(self, X, y):
        def score(estimator, X, y):
            return estimator.score(X, y)

        return self.instantiate_orphan_node(backend="none", func=score, kwargs=dict(estimator=self["model"], X=X, y=y))

    def transform(self, X):
        def transform(estimator, X):
            return estimator.transform(X)

        return self.instantiate_orphan_node(backend="none", func=transform, kwargs=dict(estimator=self["model"], X=X))

    def predict(self, X):
        def predict(estimator, X):
            return estimator.predict(X)

        return self.instantiate_orphan_node(backend="none", func=predict, kwargs=dict(estimator=self["model"], X=X))

    def decision_function(self, X):
        def decision_function(estimator, X):
            return estimator.predict_proba(X)

        return self.instantiate_orphan_node(
            backend="none", func=decision_function, kwargs=dict(estimator=self["model"], X=X),
        )

    def predict_proba(self, X):
        def predict_proba(estimator, X):
            return estimator.predict_proba(X)

        return self.instantiate_orphan_node(
            backend="none", func=predict_proba, kwargs=dict(estimator=self["model"], X=X)
        )

    def predict_log_proba(self, X):
        def predict_log_proba(estimator, X):
            return estimator.predict_proba(X)

        return self.instantiate_orphan_node(
            backend="none", func=predict_log_proba, kwargs=dict(estimator=self["model"], X=X),
        )


def instantiate_classifiers(
    features: NodeWrapper, task_name: str, split_name: str, estimator, param_grid, evaluate=False
):
    assert isinstance(features, NodeWrapper)

    assert task_name in get_ancestral_metadata(features, "tasks")
    assert split_name in get_ancestral_metadata(features, "splits")

    fold_names = get_ancestral_metadata(features, "splits")[split_name]

    task = features.graph[task_name]

    models: Dict[str, ClassifierWrapper] = dict()

    root = features.graph / task_name / split_name

    for fold_name in randomised_order(fold_names):
        split_node = root / fold_name

        # Instantiate the classifier
        model = ClassifierWrapper(
            parent=split_node,
            estimator=estimator,
            param_grid=param_grid,
            features=features,
            task=task,
            split=split_node.index.get_split_series(split=split_name, fold=fold_name),
        )

        # Dump the graph
        model.dump_graph()
        if evaluate:
            model.model.evaluate()
            model.results.evaluate()

        models[fold_name] = model

        if split_name == "deployable":
            return model

    assert len(models)

    return models
