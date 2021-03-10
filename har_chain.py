import numpy as np

from har_basic import har_basic
from src.models import random_forest
from src.models import sgd_classifier
from src.selectors import select_split
from src.selectors import select_task
from src.visualisations import umap_embedding


def predict_and_concat(key, data, **models):
    probs = {kk: model.predict_proba(data) for kk, model in models.items()}
    return np.concatenate([probs[kk] for kk in sorted(probs.keys())], axis=1)


def classifier_chain(parent, data, models):
    root = parent / f"chain_from_{sorted(models.keys())}"

    root.outputs.add_output(
        key="aggregated",
        backend="numpy",
        func=predict_and_concat,
        kwargs=dict(data=data, **models),
    )

    return root


def har_chain(
    test_dataset="anguita2013",
    fs_new=33,
    win_len=2.56,
    win_inc=1.0,
    task="har",
    split_type="predefined",
    features="ecdf",
):
    kwargs = dict(fs_new=fs_new, win_len=win_len, win_inc=win_inc, task=task, features=features,)

    dataset_alignment = dict(
        anguita2013=dict(dataset_name="anguita2013", placement="waist", modality="accel"),
        pamap2=dict(dataset_name="pamap2", placement="chest", modality="accel"),
        uschad=dict(dataset_name="uschad", placement="waist", modality="accel"),
    )

    test_dataset = dataset_alignment.pop(test_dataset)

    test_feats, test_task, test_split, test_clf = har_basic(
        split_type="predefined", **test_dataset, **kwargs
    )

    models = dict()
    for name, dataset in dataset_alignment.items():
        _, _, _, classifier = har_basic(split_type="deployable", **test_dataset, **kwargs)
        models[name] = classifier.outputs[
            list(filter(lambda kk: "model" in str(kk), classifier.outputs.keys()))[0]
        ]

    chain = classifier_chain(parent=test_feats, data=test_feats.outputs["features"], models=models)

    # Get the task (and its labels), and the train/val/test splits
    task = select_task(parent=chain, task_name=task)
    split = select_split(parent=task, split_type=split_type)

    # Learn the classifier
    clf = sgd_classifier(parent=split, split=split, task=task, data=chain)
    clf.evaluate_outputs()

    # Visualise the embeddings
    viz = umap_embedding(chain, task=task)
    viz.evaluate_outputs()

    return clf


if __name__ == "__main__":
    har_chain()
