import numpy as np

from har_basic import basic_har
from har_basic import get_classifier
from src.functional.common import sorted_node_values


def har_chain(
    test_dataset="anguita2013",
    fs_new=33,
    win_len=3,
    win_inc=1,
    task_name="har",
    split_name="predefined",
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
    feats, models = basic_har(split_name="predefined", **test_dataset, **kwargs)

    # Build a dictionary of the two source datasets
    models = {
        name: basic_har(split_name="deployable", **dataset, **kwargs)[-1] for name, dataset in dataset_alignment.items()
    }
    probs = {key: model.predict_proba(feats) for key, model in models.items()}

    graph = feats.graph / ("chained-from-" + "-".join(sorted(dataset_alignment.keys())))
    probs_as_feats = graph.instantiate_node(
        key="features", func=np.concatenate, args=[[feats] + sorted_node_values(probs)], kwargs=dict(axis=1)
    )

    # Learn the classifier
    model = get_classifier(
        clf_name=clf_name, features=probs_as_feats, task_name=task_name, split_name=split_name, evaluate=evaluate
    )

    return probs_as_feats, model


if __name__ == "__main__":
    har_chain(evaluate=True)
