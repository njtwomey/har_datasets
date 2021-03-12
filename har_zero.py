import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from har_basic import har_basic
from src.models.sklearn.base import sklearn_model_factory
from src.selectors import select_split
from src.selectors import select_task


CLASSIFIERS = dict()


class ZeroShotModel(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        self.classifiers = kwargs
        self.classes_ = None
        self.label_map = dict(
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

    def fit(self, x, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, data):
        weights = dict(zip(sorted(CLASSIFIERS.keys()), [self.alpha, 1 - self.alpha]))

        probs = {kk: pd.DataFrame(mm.predict_proba(data), columns=mm.classes_) for kk, mm in CLASSIFIERS.items()}

        label_idx = dict(zip(self.classes_, np.arange(len(self.classes_))))
        output = np.zeros((len(data), len(self.classes_)))
        for kk, prob in probs.items():
            cols = np.intersect1d(prob.columns, list(self.label_map.keys()))
            prob = prob[cols]
            prob = weights[kk] * prob / prob.values.sum(axis=1, keepdims=True)
            for col in prob.columns:
                if col in self.label_map:
                    output[:, label_idx[self.label_map[col]]] += prob[col].values

        return output

    def score(self, x, y):
        return np.mean(self.predict(x) == y)

    def predict(self, data):
        return self.classes_[self.predict_proba(data).argmax(axis=1)]


def make_zero_shot_model(parent, data, models):
    for kk, vv in models.items():
        CLASSIFIERS[kk] = vv.evaluate()
    return sklearn_model_factory(
        name=f"zero_shot_model_from={sorted(models.keys())}",
        parent=parent,
        data=data,
        model=ZeroShotModel(alpha=0.5, **CLASSIFIERS),
        xval=dict(alpha=np.linspace(0, 1, 21)),
        **models,
    )


def har_zero(
    test_dataset="anguita2013",
    fs_new=33,
    win_len=2.56,
    win_inc=1,
    task="har",
    split_type="predefined",
    features="statistical",
):
    kwargs = dict(fs_new=fs_new, win_len=win_len, win_inc=win_inc, task=task, features=features, viz=False)

    dataset_alignment = dict(
        anguita2013=dict(dataset_name="anguita2013", placement="waist", modality="accel"),
        pamap2=dict(dataset_name="pamap2", placement="chest", modality="accel"),
        uschad=dict(dataset_name="uschad", placement="waist", modality="accel"),
    )

    test_dataset = dataset_alignment.pop(test_dataset)

    test_feats, test_task, test_split, test_clf = har_basic(split_type="predefined", **test_dataset, **kwargs)

    models = dict()
    for name, dataset in dataset_alignment.items():
        _, _, _, classifier = har_basic(split_type="deployable", **dataset, **kwargs)
        models[name] = classifier.model

    task = select_task(parent=test_feats.parent, task_name=task)
    split = select_split(parent=task, split_type=split_type)

    # Learn the classifier
    clf = make_zero_shot_model(parent=split, data=test_feats, models=models)
    clf.dump_graph()
    clf.evaluate_outputs()

    return clf


if __name__ == "__main__":
    har_zero()
