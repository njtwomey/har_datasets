import numpy as np

from har_basic import basic_har
from src.base import get_ancestral_metadata
from src.motifs.models import get_classifier
from src.utils.misc import randomised_order


def har_chain(
    test_dataset="anguita2013",
    fs_new=33,
    win_len=3,
    win_inc=1,
    task_name="har",
    data_partition="predefined",
    feat_name="ecdf",
    clf_name="sgd",
    evaluate=False,
):
    # Make metadata for the experiment
    kwargs = dict(
        fs_new=fs_new, win_len=win_len, win_inc=win_inc, task_name=task_name, feat_name=feat_name, clf_name=clf_name
    )

    dataset_alignment = dict(
        anguita2013=dict(dataset_name="anguita2013", location="waist", modality="accel"),
        pamap2=dict(dataset_name="pamap2", location="chest", modality="accel"),
        uschad=dict(dataset_name="uschad", location="waist", modality="accel"),
    )

    # Extract the representation for the test dataset
    test_dataset = dataset_alignment.pop(test_dataset)
    features, test_models = basic_har(data_partition="predefined", **test_dataset, **kwargs)

    # Instantiate the root directory
    root = features.graph / f"chained-from-{sorted(dataset_alignment.keys())}"

    # Build up the list of models from aux datasets
    auxiliary_models = {train_test_split: [model] for train_test_split, model in test_models.items()}
    for model_name, model_kwargs in dataset_alignment.items():
        aux_features, aux_models = basic_har(data_partition="deployable", **model_kwargs, **kwargs)
        for fi, mi in aux_models.items():
            auxiliary_models[fi].append(mi)

    models = dict()

    # Perform the chaining
    train_test_splits = get_ancestral_metadata(features, "data_partitions")[data_partition]
    for train_test_split in randomised_order(train_test_splits):
        aux_probs = [features] + [model.predict_proba(features) for model in auxiliary_models[train_test_split]]
        prob_features = root.instantiate_orphan_node(func=np.concatenate, args=[aux_probs], kwargs=dict(axis=1))

        models[train_test_split] = get_classifier(
            clf_name=clf_name,
            feature_node=prob_features,
            task_name=task_name,
            data_partition=data_partition,
            train_test_split=train_test_split,
            evaluate=evaluate,
        )

    return features, models


if __name__ == "__main__":
    har_chain(evaluate=True)
