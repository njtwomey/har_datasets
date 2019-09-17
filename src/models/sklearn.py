from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV

from .base import ModelBase

__all__ = [
    'scale_log_reg'
]


def learn_sklearn_model(key, index, label, fold, data, model, **kwargs):
    assert index.shape[0] == label.shape[0]
    assert index.shape[0] == fold.shape[0]
    assert index.shape[0] == data.shape[0]
    # TODO/FIXME: Need to think about how to incorporate the multi-fold and multi-label aspects
    tr_inds = fold.fold_0 < 0
    x_train, y_train = data[tr_inds], label.track_0[tr_inds].values.ravel()
    model = model(**kwargs)
    model.fit(x_train, y_train)
    return model


class sklearn_model(ModelBase):
    def __init__(self, name, parent, model):
        super(sklearn_model, self).__init__(
            name=name,
            parent=parent,
            model=model,
        )
        
        label = self.index['label']
        index = self.index['index']
        fold = self.index['fold']
        
        kwargs = dict(
            model=model,
            **(self.meta['params'] or dict())
        )
        
        for key, node in self.parent.outputs.items():
            self.outputs.add_output(
                key=key,
                func=learn_sklearn_model,
                sources=dict(
                    label=label,
                    index=index,
                    fold=fold,
                    data=node,
                ),
                backend='sklearn',
                **kwargs
            )


class scale_log_reg(sklearn_model):
    def __init__(self, parent, Cs=None):
        super(scale_log_reg, self).__init__(
            name=self.__class__.__name__,
            parent=parent,
            model=lambda **kwargs: Pipeline((
                ('scale', StandardScaler()),
                ('clf', LogisticRegressionCV(**kwargs))
            ))
        )
