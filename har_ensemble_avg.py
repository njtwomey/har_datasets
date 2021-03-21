from collections import defaultdict
from typing import DefaultDict
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from mldb import NodeWrapper

from har_basic import get_classifier
from har_basic import get_features
from har_basic import get_windowed_wearables
from src.base import ExecutionGraph
from src.evaluation.classification import evaluate_data_split
from src.functional.common import node_itemgetter
from src.functional.common import sorted_node_values
from src.models.base import ClassifierWrapper
from src.utils.loaders import dataset_importer
from src.utils.misc import randomised_order
from src.visualisations.umap_embedding import umap_embedding


def ensemble_classifier(task_name, split_name, feat_names, clf_names, windowed_data, viz=False):
    model_dict: DefaultDict[str, List[Tuple[str, ClassifierWrapper]]] = defaultdict(list)
    pred_dict: DefaultDict[str, List[Tuple[str, NodeWrapper]]] = defaultdict(list)

    graph: ExecutionGraph = windowed_data / f"{sorted(feat_names)}x{sorted(clf_names)}"

    # Iterate over all features and classifiers
    for feat_name in randomised_order(feat_names):
        features = get_features(feat_name=feat_name, windowed_data=windowed_data)
        for clf_name in randomised_order(clf_names):
            model_nodes = get_classifier(
                clf_name=clf_name, features=features, task_name=task_name, split_name=split_name
            )
            for fold, estimator in model_nodes.items():
                key = f"{feat_name=}-{clf_name=}"
                model_dict[fold].append((key, model_nodes[fold].model))
                pred_dict[fold].append((key, model_nodes[fold].predict_proba(features)))

    # For each train/val/test split,
    for fold_name in model_dict.keys():
        fold = graph / fold_name

        mean_proba = fold.instantiate_node(
            key="features", func=np.mean, args=[sorted_node_values(dict(pred_dict[fold_name]))], kwargs=dict(axis=0)
        )

        if viz:
            umap_embedding(mean_proba, task_name=task_name).evaluate()

        results = fold.instantiate_node(
            key="results",
            func=evaluate_data_split,
            backend="json",
            kwargs=dict(
                split=fold.instantiate_orphan_node(func=node_itemgetter(fold_name), args=windowed_data[split_name]),
                targets=windowed_data[task_name],
                estimator=model_dict[fold_name][0][1],
                prob_predictions=mean_proba,
            ),
        )

        fold.dump_graph()

        results.evaluate()

    return model_dict


def basic_ensemble(
    dataset_name="anguita2013",
    modality="all",
    location="all",
    task_name="har",
    split_name="predefined",
    fs_new=33,
    win_len=3,
    win_inc=1,
):
    dataset = dataset_importer(dataset_name)

    windowed_data = get_windowed_wearables(
        dataset=dataset, modality=modality, location=location, fs_new=fs_new, win_len=win_len, win_inc=win_inc
    )

    ensemble = ensemble_classifier(
        task_name=task_name,
        split_name=split_name,
        feat_names=["ecdf", "statistical"],
        clf_names=["sgd", "lr", "rf"],
        windowed_data=windowed_data,
    )

    return ensemble


if __name__ == "__main__":
    basic_ensemble()
