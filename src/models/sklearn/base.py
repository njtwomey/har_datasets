from sklearn import clone
from sklearn.model_selection import GridSearchCV, GroupKFold

from src.models.base import ModelBase
from src.models.perf_evaluation import classification_perf_metrics

__all__ = [
    'sklearn_model',
]


def evaluate_performance(key, fold, label, data, models):
    res = dict()
    for fold_id in fold.columns:
        res[fold_id] = dict()
        model = models[fold_id]
        for tr_val_te in fold[fold_id].unique():
            inds = fold[fold_id] == tr_val_te
            xx, yy = data[inds], label.track_0[inds].values.ravel()
            y_hat = model.predict(xx)
            res[fold_id][tr_val_te] = classification_perf_metrics(
                yy=yy, model=model, y_hat=y_hat
            )
            if hasattr(model, 'cv_results_'):
                res[fold_id][tr_val_te]['xval'] = model.cv_results_
    return res


def _learn_sklearn_model(key, index, label, fold, data, model, n_splits):
    # TODO/FIXME: currently only using one track. Revisit later.
    tr_inds = fold == 'train'
    x_train, y_train = data[tr_inds], label.track_0[tr_inds].values.ravel()
    model = clone(model)
    if isinstance(model, GridSearchCV):
        cv = GroupKFold(n_splits=n_splits).split(x_train, y_train, index.trial.values[tr_inds])
        model.cv = list(cv)
    model.fit(x_train, y_train)
    return model


def learn_sklearn_model(key, index, label, fold, data, model, n_splits):
    assert index.shape[0] == label.shape[0]
    assert index.shape[0] == fold.shape[0]
    assert index.shape[0] == data.shape[0]
    
    models = dict()
    
    for fold_id in fold.columns:
        models[fold_id] = _learn_sklearn_model(
            key=key, index=index, fold=fold[fold_id],
            label=label, data=data, model=model,
            n_splits=n_splits,
        )
    
    return models


class sklearn_model(ModelBase):
    def __init__(self, name, parent, model, n_splits=5, save_model=True):
        super(sklearn_model, self).__init__(
            name=name,
            parent=parent,
            model=model,
        )
        
        label = self.index['label']
        index = self.index['index']
        fold = self.index['fold']
        xval = self.meta['xval']
        
        if not isinstance(model, GridSearchCV):
            model = GridSearchCV(
                estimator=model,
                param_grid=xval,
                refit=True,
                verbose=10,
            )
        
        kwargs = dict(
            model=model,
            n_splits=n_splits,
        )
        
        for key, node in self.parent.outputs.items():
            clf = self.outputs.add_output(
                key=key,
                func=learn_sklearn_model,
                label=label,
                index=index,
                fold=fold,
                data=node,
                backend=['none', 'sklearn'][save_model],
                **kwargs,
            )
            
            self.outputs.add_output(
                key=('results',) + key,
                func=evaluate_performance,
                fold=fold,
                label=label,
                data=node,
                models=clf,
                backend='json',
            )
