from sklearn.metrics import classification

from src.utils.logger import get_logger
from src.evaluation.base import EvaluationBase

logger = get_logger(__name__)

__all__ = [
    'classification_metrics'
]


class classification_metrics(EvaluationBase):
    def __init__(self, parent, *args, **kwargs):
        super(classification_metrics, self).__init__(
            name=self.__class__.__name__, parent=parent, *args, **kwargs
        )
        
        for key, node in parent.outputs.items():
            self.parent.outputs.add_output(
                key=key + ('results',),
                fold=self.index.fold,
                label=self.index[node.kwargs['task']],
                fold_id=node.kwargs['fold_id'],
                task=node.kwargs['task'],
                data=parent.parent.outputs['all'],
                model=node,
                func=evaluate_performance,
                backend='json',
            )


def evaluate_fold(key, fold, targets, predictions, model):
    res = dict()
    for tr_val_te in fold.unique():
        inds = fold == tr_val_te
        yy, pp = targets[inds], predictions[inds]
        res[tr_val_te] = dict()
        res[tr_val_te] = _classification_perf_metrics(
            labels=yy, model=model, predictions=pp
        )
        if hasattr(model, 'cv_results_'):
            res[tr_val_te]['xval'] = model.cv_results_
    return res


def evaluate_performance(key, fold, fold_id, task, label, data, model):
    res = dict()
    res[fold_id] = dict()
    for tr_val_te in fold[fold_id].unique():
        inds = fold[fold_id] == tr_val_te
        xx, yy = data[inds], label.target[inds].values.ravel()
        y_hat = model.predict(xx)
        res[fold_id][tr_val_te] = _classification_perf_metrics(
            labels=yy, model=model, predictions=y_hat
        )
        if hasattr(model, 'cv_results_'):
            res[fold_id][tr_val_te]['xval'] = model.cv_results_
    return res


def _get_class_names(model):
    if hasattr(model, 'classes_'):
        return model.classes_
    logger.exception(TypeError(
        f'The classes member cannot be extracted from this object: {model}'
    ))


def _classification_perf_metrics(labels, model, predictions):
    cols = _get_class_names(model)
    
    res = dict(
        accuracy=classification.accuracy_score(labels, predictions),
        error=1 - classification.accuracy_score(labels, predictions),
        f1_macro=classification.f1_score(labels, predictions, average='macro'),
        f1_micro=classification.f1_score(labels, predictions, average='micro'),
        f1_weighted=classification.f1_score(labels, predictions, average='weighted'),
        precision_macro=classification.precision_score(labels, predictions, average='macro'),
        precision_micro=classification.precision_score(labels, predictions, average='micro'),
        precision_weighted=classification.precision_score(labels, predictions, average='weighted'),
        recall_macro=classification.recall_score(labels, predictions, average='macro'),
        recall_micro=classification.recall_score(labels, predictions, average='micro'),
        recall_weighted=classification.recall_score(labels, predictions, average='weighted'),
        confusion_matrix=classification.confusion_matrix(labels, predictions),
        class_names=cols,
        per_class=dict()
    )
    
    if len(cols) > 2:
        for yi, col in enumerate(cols):
            yy_i = labels == col
            y_hat_i = predictions == col
            res['per_class'][f'{col}'] = dict(
                index=yi,
                label=col,
                accuracy=classification.accuracy_score(yy_i, y_hat_i),
                error=1 - classification.accuracy_score(yy_i, y_hat_i),
                f1=classification.f1_score(yy_i, y_hat_i),
                precision=classification.precision_score(yy_i, y_hat_i),
                recall=classification.recall_score(yy_i, y_hat_i),
                confusion_matrix=classification.confusion_matrix(yy_i, y_hat_i),
            )
    
    return res
