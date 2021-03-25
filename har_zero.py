from operator import itemgetter
from typing import Dict
from typing import List
from typing import Tuple

from mldb import NodeWrapper

from har_basic import basic_har
from src.base import ExecutionGraph
from src.base import get_ancestral_metadata
from src.models.base import BasicScorer
from src.models.base import ClassifierWrapper
from src.models.ensembles import ZeroShotVotingClassifier
from src.utils.misc import randomised_order


def make_zero_shot_classifier(
    feat_name,
    features,
    estimators: List[Tuple[str, NodeWrapper]],
    task_name,
    data_partition,
    train_test_split,
    label_alignment: Dict[str, str],
    evaluate: bool = True,
):
    clf_names = sorted(map(itemgetter(0), estimators))

    graph: ExecutionGraph = features.graph / f"zero-shot-from={sorted(clf_names)}" / task_name / data_partition / train_test_split

    estimator = graph.instantiate_node(
        key=f"ZeroShotVotingClassifier-{feat_name}".lower(),
        func=ZeroShotVotingClassifier,
        kwargs=dict(estimators=estimators, voting="soft", verbose=10, label_alignment=label_alignment),
    )

    model = ClassifierWrapper(
        parent=graph,
        estimator=estimator,
        features=features,
        task=features.graph[task_name],
        split=graph.get_split_series(data_partition=data_partition, train_test_split=train_test_split),
        scorer=BasicScorer(),
        evaluate=evaluate,
    )

    return model


def har_zero(
    fs_new: float = 33,
    win_len: float = 3,
    win_inc: float = 1,
    clf_name: str = "sgd",
    task_name: str = "har",
    dataset_partition: str = "predefined",
    feat_name: str = "statistical",
    evaluate: bool = False,
):
    kwargs = dict(
        fs_new=fs_new, win_len=win_len, win_inc=win_inc, task_name=task_name, feat_name=feat_name, clf_name=clf_name
    )

    external_datasets = dict(
        pamap2=dict(dataset_name="pamap2", location="chest", modality="accel"),
        uschad=dict(dataset_name="uschad", location="waist", modality="accel"),
    )

    test_dataset = dict(dataset_name="anguita2013", location="waist", modality="accel")

    label_alignment = dict(
        cycle="walk",
        elevator_down="stand",
        elevator_up="stand",
        iron="stand",
        jump="walk",
        other="walk",
        rope_jump="walk",
        run="walk",
        sit="sit",
        sleep="lie",
        stand="stand",
        vacuum="walk",
        walk="walk",
        walk_down="walk_down",
        walk_left="walk",
        walk_nordic="walk",
        walk_right="walk",
        walk_up="walk_up",
        lie="lie",
    )

    features, test_models = basic_har(data_partition="predefined", **test_dataset, **kwargs)

    auxiliary_models = dict()
    for model_name, model_kwargs in external_datasets.items():
        aux_features, aux_models = basic_har(data_partition="deployable", **model_kwargs, **kwargs)
        auxiliary_models[model_name] = aux_models

    models = dict()

    train_test_splits = get_ancestral_metadata(features, "data_partitions")[dataset_partition]
    for train_test_split in randomised_order(train_test_splits):
        models[train_test_split] = make_zero_shot_classifier(
            estimators=[(mn, mm[train_test_split].model) for mn, mm in auxiliary_models.items()],
            feat_name=feat_name,
            features=features,
            task_name=task_name,
            train_test_split=train_test_split,
            data_partition=dataset_partition,
            evaluate=evaluate,
            label_alignment=label_alignment,
        )

    return models


if __name__ == "__main__":
    har_zero(evaluate=True)
