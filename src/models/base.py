from typing import Dict

import numpy as np
from loguru import logger
from mldb import NodeWrapper
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline

from src.base import dump_graph
from src.base import ExecutionGraph
from src.evaluation.classification import evaluate_fold
from src.selectors import select_split
from src.selectors import select_task
from src.utils.misc import randomised_order

__all__ = ["instantiate_and_fit", "ClassifierWrapper", "instantiate_classifiers"]


def instantiate_and_fit(index, fold, fold_name, X, y, estimator, n_splits=5, param_grid=None):
    assert fold.shape[0] == index.shape[0]
    assert fold.shape[0] == X.shape[0]
    assert fold.shape[0] == y.shape[0]

    fold_vals = fold[fold_name].ravel()

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
class ClassifierWrapper(BaseEstimator):
    def __init__(self, root, estimator, param_grid, fold_name):
        self.root = root
        self.estimator = estimator
        self.estimator_node = None
        self.fold_name = fold_name
        self.param_grid = param_grid

    def fit(self, X, y, fold):
        assert isinstance(X, NodeWrapper)
        assert isinstance(y, NodeWrapper)

        self.estimator_node = self.root.get_or_create(
            key="model",
            func=instantiate_and_fit,
            backend="sklearn",
            kwargs=dict(
                X=X,
                y=y,
                index=self.root.index["index"],
                fold=fold,
                fold_name=self.fold_name,
                estimator=self.estimator,
                param_grid=self.param_grid,
            ),
        )

        prob_predictions = self.predict_proba(X)
        results = self.root.get_or_create(
            key="results",
            func=evaluate_fold,
            backend="json",
            kwargs=dict(
                fold=fold,
                fold_name=self.fold_name,
                targets=y,
                estimator=self.estimator_node,
                prob_predictions=prob_predictions,
            ),
        )

        return self.estimator_node, results

    def score(self, X, y):
        def score(estimator, X, y):
            return estimator.score(X, y)

        return self.root.instantiate_orphan_node(
            backend="none", func=score, kwargs=dict(estimator=self.estimator_node, X=X, y=y)
        )

    def transform(self, X):
        def transform(estimator, X):
            return estimator.transform(X)

        return self.root.instantiate_orphan_node(
            backend="none", func=transform, kwargs=dict(estimator=self.estimator_node, X=X)
        )

    def predict(self, X):
        def predict(estimator, X):
            return estimator.predict(X)

        return self.root.instantiate_orphan_node(
            backend="none", func=predict, kwargs=dict(estimator=self.estimator_node, X=X)
        )

    def decision_function(self, X):
        def decision_function(estimator, X):
            return estimator.predict_proba(X)

        return self.root.instantiate_orphan_node(
            backend="none", func=decision_function, kwargs=dict(estimator=self.estimator_node, X=X),
        )

    def predict_proba(self, X):
        def predict_proba(estimator, X):
            return estimator.predict_proba(X)

        return self.root.instantiate_orphan_node(
            backend="none", func=predict_proba, kwargs=dict(estimator=self.estimator_node, X=X)
        )

    def predict_log_proba(self, X):
        def predict_log_proba(estimator, X):
            return estimator.predict_proba(X)

        return self.root.instantiate_orphan_node(
            backend="none", func=predict_log_proba, kwargs=dict(estimator=self.estimator_node, X=X),
        )


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


def instantiate_classifiers(features, task_name, split_name, index, estimator, param_grid, evaluate=False):
    assert isinstance(features, NodeWrapper)
    assert isinstance(index, NodeWrapper)

    task = select_task(parent=features.graph, task_name=task_name)
    target = task["target"]
    splits = select_split(parent=task, split_type=split_name)

    model_nodes: Dict[str, ClassifierWrapper] = dict()

    split_names = features.graph.get_ancestral_metadata("splits")[split_name]
    for fold_name in randomised_order(split_names):
        clf_name = get_estimator_name(estimator=estimator)

        fold_graph: ExecutionGraph = splits / clf_name / fold_name

        # Instantiate the classifier
        model_node = ClassifierWrapper(
            root=fold_graph, estimator=estimator, param_grid=param_grid, fold_name=fold_name,
        )

        # Populate the nodes
        node, res = model_node.fit(features, task["target"], splits.index["split"])

        # Dump the graph
        fold_graph.dump_graph()
        if evaluate:
            node.evaluate()
            res.evaluate()

        model_nodes[fold_name] = model_node

    if split_name == "deployable":
        model_nodes = model_node

    return task, target, splits, model_nodes
