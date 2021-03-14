from typing import Dict

from mldb import NodeWrapper
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from slugify import slugify

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

    estimator_clone = clone(estimator)

    if param_grid is not None:
        grid_search = GridSearchCV(estimator=estimator_clone, param_grid=param_grid, verbose=10)
        k_fold = GroupKFold(n_splits=n_splits).split(X[train_inds], y[train_inds], index.trial.values[train_inds])
        grid_search.cv = list(k_fold)
        return grid_search.fit(X[train_inds], y[train_inds])

    raise estimator_clone.fit(X[train_inds], y[train_inds])


# noinspection PyPep8Naming
class ClassifierWrapper(BaseEstimator):
    def __init__(self, root, estimator, param_grid, fold_name):
        self.root = root
        self.estimator_object = estimator
        self.estimator = None
        self.fold_name = fold_name
        self.param_grid = param_grid

    def fit(self, X, y, fold):
        assert isinstance(X, NodeWrapper)
        assert isinstance(y, NodeWrapper)
        self.estimator_node = self.root.outputs.get_or_create(
            key="model",
            func=instantiate_and_fit,
            backend="sklearn",
            kwargs=dict(
                X=X,
                y=y,
                index=self.root.index["index"],
                fold=fold,
                fold_name=self.fold_name,
                estimator=self.estimator_object,
                param_grid=self.param_grid,
            ),
        )
        return self.estimator_node

    def score(self, X, y):
        def score(estimator, X, y):
            return estimator.score(X, y)

        return self.root.outputs.get_or_create(
            key="score", backend="none", func=score, kwargs=dict(estimator=self.estimator_node, X=X, y=y)
        )

    def results(self, X, y, fold):
        predictions = self.predict(X)
        prob_predictions = self.predict_proba(X)
        estimator = self.fit(X=X, y=y, fold=fold)
        return self.root.outputs.create(
            key="results",
            func=evaluate_fold,
            backend="json",
            kwargs=dict(
                fold=fold,
                fold_name=self.fold_name,
                targets=y,
                estimator=estimator,
                predictions=predictions,
                prob_predictions=prob_predictions,
            ),
        )

    def transform(self, X):
        def transform(estimator, X):
            return estimator.transform(X)

        return self.root.outputs.get_or_create(
            key="transformation", backend="none", func=transform, kwargs=dict(estimator=self.estimator_node, X=X)
        )

    def predict(self, X):
        def predict(estimator, X):
            return estimator.predict(X)

        return self.root.outputs.get_or_create(
            key="predictions", backend="none", func=predict, kwargs=dict(estimator=self.estimator_node, X=X)
        )

    def decision_function(self, X):
        def decision_function(estimator, X):
            return estimator.predict_proba(X)

        return self.root.outputs.get_or_create(
            key="decision_function",
            backend="none",
            func=decision_function,
            kwargs=dict(estimator=self.estimator_node, X=X),
        )

    def predict_proba(self, X):
        def predict_proba(estimator, X):
            return estimator.predict_proba(X)

        return self.root.outputs.get_or_create(
            key="prob_predictions", backend="none", func=predict_proba, kwargs=dict(estimator=self.estimator_node, X=X)
        )

    def predict_log_proba(self, X):
        def predict_log_proba(estimator, X):
            return estimator.predict_proba(X)

        return self.root.outputs.get_or_create(
            key="log_prob_predictions",
            backend="none",
            func=predict_log_proba,
            kwargs=dict(estimator=self.estimator_node, X=X),
        )


def get_estimator_name(estimator):
    if isinstance(estimator, Pipeline):
        return [f"{name}='{slugify(inst.__class__.__name__)}'" for name, inst in sorted(estimator.steps)]
    elif isinstance(estimator, FeatureUnion):
        raise NotImplementedError
    return [slugify(estimator.__class__.__name__)]


def instantiate_classifiers(
    features, task_name, split_name, index, estimator, param_grid
) -> Dict[str, ClassifierWrapper]:
    assert isinstance(features, NodeWrapper)
    assert isinstance(index, NodeWrapper)

    task = select_task(parent=features.graph, task_name=task_name)
    splits = select_split(parent=task, split_type=split_name)

    model_nodes: Dict[str, ClassifierWrapper] = dict()

    for fold_name in randomised_order(splits["split"].evaluate()):
        clf_name = get_estimator_name(estimator=estimator)

        fold_graph = splits / "-".join(clf_name) / fold_name

        # Instantiate the classifier
        model_node = ClassifierWrapper(
            root=fold_graph, estimator=estimator, param_grid=param_grid, fold_name=fold_name,
        )

        # Populate the nodes
        model_node.fit(features, task["target"], fold_graph["fold"])
        model_node.results(features, task["target"], fold_graph["fold"])
        model_node.predict(features)
        model_node.predict_proba(features)

        # Dump the graph
        fold_graph.dump_graph()
        fold_graph.evaluate()

        model_nodes[fold_name] = model_node

    return model_nodes
