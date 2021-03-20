from collections import defaultdict

import numpy as np

from har_basic import get_classifier
from har_basic import get_features
from har_basic import get_windowed_wearables
from src import evaluate_fold
from src import ExecutionGraph
from src import randomised_order
from src.utils.loaders import dataset_importer


def sorted_vals(**kwargs):
    return [kwargs[kk] for kk in sorted(kwargs.keys())]


def mean(axis=0, **kwargs):
    return np.mean(sorted_vals(**kwargs), axis=axis)


def ensemble_classifier(task_name, split_name, feat_names, clf_names, windowed_data):
    models, preds = defaultdict(list), defaultdict(list)

    for feat_name in randomised_order(feat_names):
        features = get_features(feat_name=feat_name, windowed_data=windowed_data)
        for clf_name in randomised_order(clf_names):
            task, target, splits, model_nodes = get_classifier(
                clf_name=clf_name, features=features, task_name=task_name, split_name=split_name
            )
            for fold, estimator in model_nodes.items():
                models[fold].append((f"{feat_name=}-{clf_name=}", model_nodes[fold].estimator_node))
                preds[fold].append((f"{feat_name=}-{clf_name=}", model_nodes[fold].predict_proba(features)))

    node: ExecutionGraph = windowed_data / f"{feat_names}x{clf_names}"

    for fold_name in models.keys():
        fold = node / fold_name

        probs = fold.instantiate_orphan_node(mean, kwargs=dict(axis=0, **dict(preds[fold_name])))

        fold.instantiate_node(
            key="results",
            func=evaluate_fold,
            backend="json",
            kwargs=dict(
                fold=windowed_data["fold"],
                fold_name=fold_name,
                targets=windowed_data["har"],
                estimator=estimator.estimator_node.evaluate(),
                prob_predictions=probs,
            ),
        ).evaluate()

        fold.dump_graph()

    return models


def basic_ensemble(
    dataset_name="pamap2",
    modality="all",
    location="all",
    task_name="har",
    split_name="predefined",
    fs_new=33,
    win_len=3,
    win_inc=1,
):
    dataset = dataset_importer(dataset_name)

    windowed = get_windowed_wearables(
        dataset=dataset, modality=modality, location=location, fs_new=fs_new, win_len=win_len, win_inc=win_inc
    )

    ensemble = ensemble_classifier(
        task_name=task_name,
        split_name=split_name,
        feat_names=["ecdf", "statistical"],
        clf_names=["sgd", "lr", "rf"],
        windowed_data=windowed,
    )

    return ensemble


if __name__ == "__main__":
    basic_ensemble()
