from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from mldb import NodeWrapper
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold

from src.base import ExecutionGraph
from src.evaluation.classification import evaluate_data_split

__all__ = ["instantiate_and_fit", "ClassifierWrapper", "BasicScorer"]


def instantiate_and_fit(
    index: pd.DataFrame,
    fold: pd.DataFrame,
    X: np.ndarray,
    y: pd.DataFrame,
    estimator: BaseEstimator,
    n_splits: int = 5,
    param_grid: Optional[Dict[str, Any]] = None,
) -> BaseEstimator:
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

    logger.info(f"Fitting {estimator} on data (shape: {X.shape})")

    if param_grid is not None:
        group_k_fold = GroupKFold(n_splits=n_splits).split(X[train_inds], y[train_inds], index.trial.values[train_inds])

        grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, verbose=10, cv=list(group_k_fold))
        grid_search.fit(X[train_inds], y[train_inds])

        return grid_search.best_estimator_

    estimator.fit(X[train_inds], y[train_inds])
    return estimator


# noinspection PyPep8Naming
class BasicScorer(object):
    def fit(self, estimator: Any, X: np.ndarray, y: np.ndarray):
        return estimator.fit(X, y)

    def score(self, estimator: Any, X: np.ndarray, y: np.ndarray):
        return estimator.score(X, y)

    def transform(self, estimator: Any, X: np.ndarray):
        return estimator.transform(X)

    def decision_function(self, estimator: Any, X: np.ndarray):
        return estimator.predict_proba(X)

    def predict(self, estimator: Any, X: np.ndarray):
        return estimator.predict(X)

    def predict_proba(self, estimator: Any, X: np.ndarray):
        return estimator.predict_proba(X)

    def predict_log_proba(self, estimator: Any, X: np.ndarray):
        return estimator.predict_proba(X)


# noinspection PyPep8Naming
class ClassifierWrapper(ExecutionGraph):
    def __init__(
        self,
        parent: ExecutionGraph,
        features: NodeWrapper,
        split: NodeWrapper,
        task: NodeWrapper,
        estimator: NodeWrapper,
        param_grid: Optional[Dict[str, Any]] = None,
        scorer: Optional[BasicScorer] = None,
        evaluate: bool = False,
    ):
        assert isinstance(parent, ExecutionGraph)
        assert isinstance(features, NodeWrapper)
        assert isinstance(split, NodeWrapper)
        assert isinstance(task, NodeWrapper)
        assert isinstance(estimator, NodeWrapper)

        super(ClassifierWrapper, self).__init__(parent=parent, name=f"estimator={str(estimator.name.name)}")

        self.features = features
        self.split = split
        self.task = task

        self.scorer = BasicScorer() if scorer is None else scorer

        model = self.instantiate_node(
            key="model",
            func=instantiate_and_fit,
            backend="sklearn",
            kwargs=dict(
                X=features, y=task, index=self["index"], fold=self.split, estimator=estimator, param_grid=param_grid,
            ),
        )

        results = self.get_or_create(
            key="results",
            func=evaluate_data_split,
            backend="json",
            kwargs=dict(split=split, targets=task, estimator=model, prob_predictions=self.predict_proba(features)),
        )

        if evaluate:
            self.dump_graph()
            model.evaluate()
            results.evaluate()

    @property
    def model(self):
        return self["model"]

    @property
    def results(self):
        return self["results"]

    def fit(self, X, y) -> NodeWrapper:
        logger.warning(f"it looks like you're attempting to re-fit a model on new data - is this the intent?")
        return self.instantiate_orphan_node(func=self.scorer.fit, kwargs=dict(estimator=self["model"], X=X, y=y),)

    def score(self, X, y) -> NodeWrapper:
        return self.instantiate_orphan_node(func=self.scorer.score, kwargs=dict(estimator=self["model"], X=X, y=y))

    def transform(self, X) -> NodeWrapper:
        return self.instantiate_orphan_node(func=self.scorer.transform, kwargs=dict(estimator=self["model"], X=X))

    def predict(self, X) -> NodeWrapper:
        return self.instantiate_orphan_node(func=self.scorer.predict, kwargs=dict(estimator=self["model"], X=X))

    def decision_function(self, X) -> NodeWrapper:
        return self.instantiate_orphan_node(
            func=self.scorer.decision_function, kwargs=dict(estimator=self["model"], X=X)
        )

    def predict_proba(self, X) -> NodeWrapper:
        return self.instantiate_orphan_node(func=self.scorer.predict_proba, kwargs=dict(estimator=self["model"], X=X))

    def predict_log_proba(self, X) -> NodeWrapper:
        return self.instantiate_orphan_node(
            func=self.scorer.predict_log_proba, kwargs=dict(estimator=self["model"], X=X)
        )
