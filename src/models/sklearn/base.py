from os.path import join

import numpy as np
from sklearn import clone
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold

from src import BaseGraph
from src.evaluation.classification import evaluate_fold
from src.utils.misc import randomised_order


__all__ = [
    "sklearn_model",
]


def select_fold(folds, fold_name):
    assert fold_name in folds.columns
    fold_def = folds[fold_name]
    fold_vals = set(np.unique(fold_def.values))
    assert fold_vals.issubset({"train", "val", "test"})
    return fold_def


def learn_sklearn_model(index, features, targets, fold_def, model, n_splits):
    assert index.shape[0] == features.shape[0]
    assert index.shape[0] == targets.shape[0]
    assert index.shape[0] == fold_def.shape[0]

    tr_inds = fold_def == "train"

    x_train, y_train = features[tr_inds], targets["target"][tr_inds].values.ravel()

    model = clone(model)

    if "val" in fold_def:
        raise NotImplementedError

    else:
        if isinstance(model, GridSearchCV):
            cv = GroupKFold(n_splits=n_splits).split(x_train, y_train, index.trial.values[tr_inds])
            model.cv = list(cv)

    model.fit(x_train, y_train)

    return model


def sklearn_preds(model, features):
    return model.predict(features)


def sklearn_probs(model, features):
    return model.predict_proba(features)


def sklearn_decision_function(model, features):
    if hasattr(model, "decision_function"):
        return model.decision_function(features)
    elif hasattr(model, "predict_log_proba"):
        return model.predict_log_proba(features)
    raise ValueError


def sklearn_model(name, parent, model, xval, features, targets, split, fold_name, n_splits=5):
    root = parent / name / fold_name

    if not isinstance(model, GridSearchCV):
        model = GridSearchCV(estimator=model, param_grid=xval, refit=True, verbose=10)

    fold_definition = root.outputs.add_output(
        key="fold", func=select_fold, backend="none", kwargs=dict(folds=split, fold_name=fold_name),
    )

    model_instance = root.outputs.add_output(
        key="model",
        func=learn_sklearn_model,
        backend="sklearn",
        kwargs=dict(
            index=root.index["index"],
            features=features,
            targets=targets,
            fold_def=fold_definition,
            model=model,
            n_splits=n_splits,
        ),
    )

    model_predictions = root.outputs.add_output(
        key="preds",
        func=sklearn_preds,
        backend="none",
        kwargs=dict(features=features, model=model_instance),
    )

    scores = root.outputs.add_output(
        key="probs",
        func=sklearn_probs,
        backend="none",
        kwargs=dict(features=features, model=model_instance),
    )

    root.outputs.add_output(
        key="results",
        func=evaluate_fold,
        backend="json",
        kwargs=dict(
            fold=fold_definition,
            targets=targets,
            predictions=model_predictions,
            model=model_instance,
            scores=scores,
        ),
    )

    return root


class sklearn_model_factory(BaseGraph):
    def __init__(self, name, parent, data, model, xval, n_splits=5):
        super(sklearn_model_factory, self).__init__(name=name, parent=parent)

        del parent

        self.models = {}

        def append_model(fold_name):
            clf = sklearn_model(
                name=name,
                parent=self.parent,
                model=model,
                xval=xval,
                fold_name=fold_name,
                features=data.outputs["features"],
                targets=self.index["target"],
                split=self.index["split"],
                n_splits=n_splits,
            )

            self.models[fold_name] = clf
            self.outputs.acquire(clf.outputs)

        for fold_name in randomised_order(self.index["split"].evaluate()):
            append_model(fold_name=str(fold_name))

    @property
    def model_node(self):
        return next(iter(self.models.values())).outputs

    @property
    def model(self):
        return self.model_node["model"]

    @property
    def results(self):
        return self.model_node["results"]
