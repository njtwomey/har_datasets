from sklearn.metrics import classification

from src.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    'classification_perf_metrics'
]


def get_class_names(model):
    if hasattr(model, 'classes_'):
        return model.classes_
    logger.exception(f'The classes member cannot be extracted from this object: {model}')
    raise TypeError


def classification_perf_metrics(yy, model, y_hat):
    cols = get_class_names(model)
    
    res = dict(
        accuracy=classification.accuracy_score(yy, y_hat),
        error=1 - classification.accuracy_score(yy, y_hat),
        f1_macro=classification.f1_score(yy, y_hat, average='macro'),
        f1_micro=classification.f1_score(yy, y_hat, average='micro'),
        f1_weighted=classification.f1_score(yy, y_hat, average='weighted'),
        precision_macro=classification.precision_score(yy, y_hat, average='macro'),
        precision_micro=classification.precision_score(yy, y_hat, average='micro'),
        precision_weighted=classification.precision_score(yy, y_hat, average='weighted'),
        recall_macro=classification.recall_score(yy, y_hat, average='macro'),
        recall_micro=classification.recall_score(yy, y_hat, average='micro'),
        recall_weighted=classification.recall_score(yy, y_hat, average='weighted'),
        confusion_matrix=classification.confusion_matrix(yy, y_hat),
        class_names=cols,
        per_class=dict()
    )
    
    if len(cols) > 2:
        for yi, col in enumerate(cols):
            yy_i = yy == col
            y_hat_i = y_hat == col
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
