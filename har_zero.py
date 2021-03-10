from collections import Counter

import numpy as np
import pandas as pd

from har_basic import har_basic
from src.evaluation.classification import evaluate_fold


def resolve_zero_shot(data, label_names, targets, **models):
    label_names = np.asarray(label_names)

    probs = {
        kk: pd.DataFrame(mm.predict_proba(data), columns=mm.classes_) for kk, mm in models.items()
    }

    label_map = dict(
        # cycle="walk",
        elevator_down="stand",
        elevator_up="stand",
        iron="stand",
        # jump="walk",
        # other="walk",
        # rope_jump="walk",
        run="walk",
        sit="sit",
        sleep="lie",
        stand="stand",
        # vacuum="walk",
        walk="walk",
        walk_down="walk_down",
        walk_left="walk",
        walk_nordic="walk",
        walk_right="walk",
        walk_up="walk_up",
        lie="lie",
    )

    output = np.zeros((len(data), len(label_names)))
    for kk, df in probs.items():
        for col in df.columns:
            if col in label_map:
                output[:, list(label_names).index(label_map[col])] += df[col].values

    predictions = np.asarray(label_names)[output.argmax(1)]

    class Model:
        def __init__(self):
            self.classes_ = label_names

    mets = evaluate_fold(
        fold=pd.Series(np.repeat("test", data.shape[0])),
        targets=targets,
        predictions=predictions,
        scores=output,
        model=Model(),
    )

    return mets


def zero_shot(parent, data, label_names, models, targets):
    root = parent / f"zero_shot_from_{sorted(models.keys())}"

    root.outputs.add_output(
        key="results",
        backend="json",
        func=resolve_zero_shot,
        kwargs=dict(data=data, label_names=label_names, targets=targets, **models),
    )

    return root


def har_zero(
    test_dataset="anguita2013",
    fs_new=33,
    win_len=2.56,
    win_inc=1,
    task="har",
    split_type="predefined",
    features="ecdf",
):
    kwargs = dict(fs_new=fs_new, win_len=win_len, win_inc=win_inc, task=task, features=features)

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
        _, _, _, classifier = har_basic(split_type="deployable", **dataset, **kwargs)
        models[name] = classifier.model

    test_labels = list(test_feats.get_ancestral_metadata("tasks")["har"]["target_transform"].keys())

    zero = zero_shot(
        parent=test_feats,
        data=test_feats.outputs["features"],
        label_names=test_labels,
        models=models,
        targets=test_feats.index["har"],
    )
    zero.evaluate_outputs()

    return zero


if __name__ == "__main__":
    har_zero()
