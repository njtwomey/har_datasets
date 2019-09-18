from sklearn import clone

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from .base import ModelBase
from .perf_evaluation import classification_perf_metrics

__all__ = [
    'scale_log_reg',
    'random_forest',
    'knn',
    'svm',
]


def evaluate_performance(key, fold, label, data, models):
    res = dict()
    for fold_id in fold.columns:
        res[fold_id] = dict()
        model = models[fold_id]
        for tr_val_te in fold[fold_id].unique():
            inds = fold[fold_id] == tr_val_te
            xx, yy = data[inds], label.track_0[inds].values.ravel()
            # pp = model.predict_proba(xx)
            y_hat = model.predict(xx)
            res[fold_id][tr_val_te] = classification_perf_metrics(
                yy=yy, model=model, y_hat=y_hat
            )
    return res


def _learn_sklearn_model(key, index, label, fold, data, model):
    tr_inds = fold == 'train'
    # TODO/FIXME: currently only using one track. Revisit later.
    x_train, y_train = data[tr_inds], label.track_0[tr_inds].values.ravel()
    model = clone(model)
    model.fit(x_train, y_train)
    return model


def learn_sklearn_model(key, index, label, fold, data, model):
    assert index.shape[0] == label.shape[0]
    assert index.shape[0] == fold.shape[0]
    assert index.shape[0] == data.shape[0]
    models = dict()
    for fold_id in fold.columns:
        models[fold_id] = _learn_sklearn_model(
            key=key, index=index, fold=fold[fold_id], label=label,
            data=data, model=model,
        )
    return models


class sklearn_model(ModelBase):
    def __init__(self, name, parent, model, save_model=True):
        super(sklearn_model, self).__init__(
            name=name,
            parent=parent,
            model=model,
        )
        
        label = self.index['label']
        index = self.index['index']
        fold = self.index['fold']
        
        if 'xval' in self.meta:
            model = GridSearchCV(
                estimator=model(),
                param_grid=self.meta['xval'],
                refit=True,
                verbose=10,
            )
        else:
            model = model(**self.meta['params'])
            
        kwargs = dict(
            model=model,
        )
        
        for key, node in self.parent.outputs.items():
            clf = self.outputs.add_output(
                key=key,
                func=learn_sklearn_model,
                sources=dict(
                    label=label,
                    index=index,
                    fold=fold,
                    data=node,
                ),
                backend=['none', 'sklearn'][save_model],
                **kwargs,
            )
            
            self.outputs.add_output(
                key=('results',) + key,
                func=evaluate_performance,
                sources=dict(
                    fold=fold,
                    label=label,
                    data=node,
                    models=clf,
                ),
                backend='json',
            )


class scale_log_reg(sklearn_model):
    def __init__(self, parent):
        super(scale_log_reg, self).__init__(
            name=self.__class__.__name__,
            parent=parent,
            model=lambda **kwargs: Pipeline((
                ('scale', StandardScaler()),
                ('clf', LogisticRegressionCV(**kwargs))
            ))
        )


class random_forest(sklearn_model):
    def __init__(self, parent):
        super(random_forest, self).__init__(
            name=self.__class__.__name__,
            parent=parent,
            model=lambda **kwargs: RandomForestClassifier(**kwargs)
        )


class knn(sklearn_model):
    def __init__(self, parent):
        super(knn, self).__init__(
            name=self.__class__.__name__,
            parent=parent,
            model=lambda **kwargs: Pipeline((
                ('scale', StandardScaler()),
                ('clf', KNeighborsClassifier(**kwargs))
            )),
            save_model=False,
        )


class svm(sklearn_model):
    def __init__(self, parent):
        super(svm, self).__init__(
            name=self.__class__.__name__,
            parent=parent,
            model=lambda **kwargs: Pipeline((
                ('scale', MinMaxScaler()),
                ('clf', SVC(**kwargs))
            )),
            save_model=False,
        )
