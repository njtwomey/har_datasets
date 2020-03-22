from sklearn import clone
from sklearn.model_selection import GridSearchCV, GroupKFold

import pandas as pd
import numpy as np

from os.path import join

from src.models.base import ModelBase
from src.utils.logger import get_logger
from src.utils.misc import randomised_order
from src.evaluation.classification import evaluate_fold

logger = get_logger(__name__)

__all__ = [
    'sklearn_model',
]


def select_fold(key, folds, fold_name):
    assert fold_name in folds.columns
    fold_def = folds[fold_name]
    fold_vals = set(np.unique(fold_def.values))
    assert fold_vals.issubset({'train', 'val', 'test'})
    return fold_def


def learn_sklearn_model(key, index, features, targets, fold_def, model, n_splits):
    assert index.shape[0] == features.shape[0]
    assert index.shape[0] == targets.shape[0]
    assert index.shape[0] == fold_def.shape[0]

    tr_inds = fold_def == 'train'

    x_train, y_train = features[tr_inds], targets['target'][tr_inds].values.ravel()

    model = clone(model)

    if 'val' in fold_def:
        raise NotImplementedError

    else:
        if isinstance(model, GridSearchCV):
            cv = GroupKFold(n_splits=n_splits).split(x_train, y_train, index.trial.values[tr_inds])
            model.cv = list(cv)

    model.fit(x_train, y_train)

    return model


def sklearn_preds(key, model, features):
    return model.predict(features)


def sklearn_probs(key, model, features):
    return model.predict_proba(features)


class sklearn_model(ModelBase):
    def __init__(self, name, parent, model, xval, features, targets, split, fold_name, n_splits=5):
        super(sklearn_model, self).__init__(
            name=name, parent=parent, model=model,
        )

        if not isinstance(model, GridSearchCV):
            model = GridSearchCV(
                estimator=model,
                param_grid=xval,
                refit=True,
                verbose=10,
            )

        fold_def = self.outputs.add_output(
            key=join(fold_name, 'fold'),
            func=select_fold,
            backend='none',
            folds=split,
            fold_name=fold_name,
        )

        model = self.outputs.add_output(
            key=join(fold_name, 'model'),
            func=learn_sklearn_model,
            index=self.index['index'],
            features=features,
            targets=targets,
            fold_def=fold_def,
            model=model,
            n_splits=n_splits,
            backend='sklearn',
        )

        predictions = self.outputs.add_output(
            key=join(fold_name, 'preds'),
            func=sklearn_preds,
            backend='none',
            features=features,
            model=model,
        )

        self.outputs.add_output(
            key=join(fold_name, 'probs'),
            func=sklearn_probs,
            backend='none',
            features=features,
            model=model,
        )

        self.outputs.add_output(
            key=join(fold_name, 'results'),
            func=evaluate_fold,
            backend='json',
            fold=fold_def,
            targets=targets,
            predictions=predictions,
            model=model,
        )


class sklearn_model_factory(ModelBase):
    def __init__(self, name, parent, data, model, xval, n_splits=5):
        super(sklearn_model_factory, self).__init__(
            name=name, parent=parent, model=model,
        )

        del parent

        self.models = {}

        def append_model(fold_name):
            clf = sklearn_model(
                name=name,
                parent=self.parent,
                model=model,
                xval=xval,
                fold_name=fold_name,
                features=data.outputs['features'],
                targets=self.index['target'],
                split=self.index['split'],
                n_splits=n_splits,
            )

            self.models[fold_name] = clf
            self.outputs.acquire(clf.outputs)

        for fold_name in randomised_order(self.index['split'].evaluate()):
            append_model(fold_name=str(fold_name))
