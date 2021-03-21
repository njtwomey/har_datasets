from collections import Counter

import numpy as np
from loguru import logger
from scipy.special import logsumexp
from sklearn import metrics


__all__ = ["evaluate_data_split"]


def evaluate_data_split(split, targets, estimator, prob_predictions):
    res = dict()
    predictions = estimator.classes_[prob_predictions.argmax(axis=1)]
    for tr_val_te in split.unique():
        inds = split == tr_val_te
        yy, pp, ss = targets[inds], predictions[inds], prob_predictions[inds]
        res[tr_val_te] = dict()
        res[tr_val_te] = _classification_perf_metrics(
            model=estimator, labels=np.asarray(yy).ravel(), predictions=np.asarray(pp), scores=np.asarray(ss),
        )
    if hasattr(estimator, "cv_results_"):
        res["xval"] = estimator.cv_results_
    return res


def _get_class_names(model):
    if hasattr(model, "classes_"):
        return model.classes_
    logger.exception(TypeError(f"The classes member cannot be extracted from this object: {model}"))


def _classification_perf_metrics(labels, model, predictions, scores):
    cols = _get_class_names(model)

    def score_metrics(name, func, labels_, scores_, **kwargs):
        unique_labels = np.unique(labels_)
        lookup = dict(zip(unique_labels, range(unique_labels.shape[0])))
        scores_ = scores_[:, unique_labels]
        if scores_.shape[1] == 1:
            return 1.0
        scores_ /= scores_.sum(axis=1, keepdims=True)
        if scores_.shape[1] == 2:
            scores_ = scores_[:, 1]
        return {
            f"{name}_{average}": func(
                y_true=np.asarray([lookup[label] for label in labels_]), y_score=scores_, average=average, **kwargs,
            )
            for average in ("macro", "weighted")
        }

    def prediction_metrics(name, func):
        return {
            f"{name}_{average}": func(y_true=labels, y_pred=predictions, average=average)
            for average in ("macro", "micro", "weighted")
        }

    label_lookup = dict(zip(model.classes_, range(model.classes_.shape[0])))
    probs = np.exp(scores - logsumexp(scores, axis=1, keepdims=True))
    label_ind = np.asarray([label_lookup[ll] for ll in labels])

    label_counts = Counter(labels)

    res = dict(
        class_names=cols,
        num_instances=len(labels),
        label_counts=dict(label_counts.items()),
        class_prior={kk: vv / len(labels) for kk, vv in label_counts.items()},
        accuracy=metrics.accuracy_score(labels, predictions),
        confusion_matrix=metrics.confusion_matrix(labels, predictions),
        **score_metrics("auroc_ovo", metrics.roc_auc_score, label_ind, probs, multi_class="ovo"),
        **score_metrics("auroc_ovr", metrics.roc_auc_score, label_ind, probs, multi_class="ovr"),
        **prediction_metrics("f1", metrics.f1_score),
        **prediction_metrics("precision", metrics.precision_score),
        **prediction_metrics("recall", metrics.recall_score),
        per_class_metrics=dict(),
    )

    if len(cols) > 2:
        for col in np.unique(labels):
            yi = label_lookup[col]
            yy_i = labels == col
            y_hat_i = predictions == col
            res["per_class_metrics"][col] = dict(
                index=yi,
                label=col,
                count=yy_i.sum(),
                class_prior=yy_i.mean(),
                accuracy=metrics.accuracy_score(yy_i, y_hat_i),
                auroc=metrics.roc_auc_score(yy_i, y_hat_i),
                f1=metrics.f1_score(yy_i, y_hat_i),
                precision=metrics.precision_score(yy_i, y_hat_i),
                recall=metrics.recall_score(yy_i, y_hat_i),
                confusion_matrix=metrics.confusion_matrix(yy_i, y_hat_i),
            )

    return res
