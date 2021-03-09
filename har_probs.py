import numpy as np

from har_basic import har_basic
from src import BaseGraph
from src.features import ecdf
from src.features import statistical_features
from src.models import sgd_classifier
from src.selectors import select_split
from src.selectors import select_task
from src.transformers import body_grav_filter
from src.transformers import resample
from src.transformers import window
from src.transformers.modality_selector import modality_selector
from src.utils.loaders import dataset_importer
from src.visualisations import umap_embedding


def predict_and_concat(key, data, **models):
    probs = {kk: model.predict_proba(data) for kk, model in models.items()}
    return np.concatenate([probs[kk] for kk in sorted(probs.keys())], axis=1)


class ClassifierChain(BaseGraph):
    def __init__(self, parent, data, classifiers):
        super(ClassifierChain, self).__init__(
            name=f"chain_from_{sorted(classifiers.keys())}", parent=parent,
        )

        models = {
            kk: classifiers[kk].outputs[
                list(filter(lambda kk: "model" in str(kk), classifiers[kk].outputs.keys()))[0]
            ]
            for kk in sorted(classifiers.keys())
        }

        self.outputs.add_output(
            key="aggregated",
            backend="numpy",
            func=predict_and_concat,
            kwargs=dict(data=data, **models),
        )


def har_prob_features(
    test_dataset="anguita2013",
    fs_new=33,
    win_len=2.56,
    win_inc=1.0,
    task="har",
    split_type="deployable",
    features="ecdf",
    modality="accel",
    location="all",
):
    test_feats, test_task, test_split, test_clf = har_basic(
        dataset_name=test_dataset, split_type="predefined", placement="waist", modality="accel"
    )

    _, _, _, pamap_clf = har_basic(
        dataset_name="pamap2", split_type="deployable", placement="chest", modality="accel"
    )
    _, _, _, uschad_clf = har_basic(
        dataset_name="uschad", split_type="deployable", placement="waist", modality="accel"
    )

    chain = ClassifierChain(
        parent=test_feats,
        data=test_feats.outputs["features"],
        classifiers=dict(pamap2=pamap_clf, uschad=uschad_clf),
    )

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
    har_prob_features()
