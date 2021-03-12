import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from har_basic import har_basic
from src.models.sklearn.base import sklearn_model_factory
from src.selectors import select_split
from src.selectors import select_task


CLASSIFIERS = dict()


def chain(data):
    probs = {kk: vv.predict_proba(data) for kk, vv in CLASSIFIERS.items()}
    return np.concatenate([probs[kk] for kk in sorted(probs.keys())], axis=1)


class ClassifierChain(SGDClassifier):
    def fit(self, x, y):
        return super(ClassifierChain, self).fit(chain(x), y)

    def predict_proba(self, x):
        return super(ClassifierChain, self).predict_proba(chain(x))

    def predict(self, x):
        return super(ClassifierChain, self).predict(chain(x))


def classifier_chain(parent, data_node, model_nodes):
    for kk, vv in model_nodes.items():
        CLASSIFIERS[kk] = vv.evaluate()
    return sklearn_model_factory(
        name=f"classifier_chain={sorted(model_nodes.keys())}",
        parent=parent,
        data=data_node,
        model=Pipeline([("scale", StandardScaler()), ("clf", ClassifierChain())]),
        xval=dict(clf__loss=["log"], clf__penalty=["l2"], clf__alpha=np.power(10.0, np.arange(-5, 5 + 1)),),
        **model_nodes,
    )


def har_chain(
    test_dataset="anguita2013",
    fs_new=33,
    win_len=2.56,
    win_inc=1,
    task="har",
    split_type="predefined",
    features="ecdf",
):
    # Make metadata for the experiment
    kwargs = dict(fs_new=fs_new, win_len=win_len, win_inc=win_inc, task=task, features=features)
    dataset_alignment = dict(
        anguita2013=dict(dataset_name="anguita2013", placement="waist", modality="accel"),
        pamap2=dict(dataset_name="pamap2", placement="chest", modality="accel"),
        uschad=dict(dataset_name="uschad", placement="waist", modality="accel"),
    )

    # Extract the representation for the test datasett
    test_dataset = dataset_alignment.pop(test_dataset)
    test_feats, test_task, test_split, test_clf = har_basic(split_type="predefined", **test_dataset, **kwargs)

    # Build a dictionary of the two source datasets
    models = {
        name: har_basic(split_type="deployable", **dataset, **kwargs)[-1].model
        for name, dataset in dataset_alignment.items()
    }

    # Get the task (and its labels), and the train/val/test splits
    task = select_task(parent=test_feats.parent, task_name=task)
    split = select_split(parent=task, split_type=split_type)

    # Learn the classifier
    clf = classifier_chain(parent=split, data_node=test_feats, model_nodes=models)
    clf.dump_graph()
    clf.evaluate_outputs()

    return clf


if __name__ == "__main__":
    har_chain()
