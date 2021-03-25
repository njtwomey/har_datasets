from mldb import NodeWrapper

from src.base import ExecutionGraph
from src.base import get_ancestral_metadata
from src.models.base import BasicScorer
from src.models.base import ClassifierWrapper
from src.models.ensembles import PrefittedVotingClassifier
from src.motifs.features import get_features
from src.motifs.features import get_windowed_wearables
from src.motifs.models import get_classifier
from src.utils.loaders import dataset_importer
from src.utils.misc import randomised_order


def ensemble_classifier(
    task_name: str,
    feat_name: str,
    data_partition: str,
    train_test_split: str,
    windowed_data: NodeWrapper,
    clf_names,
    evaluate=False,
):
    features = get_features(feat_name=feat_name, windowed_data=windowed_data)

    graph: ExecutionGraph = features.graph / f"ensemble-over={sorted(clf_names)}" / task_name / train_test_split

    estimators = list()
    for clf_name in randomised_order(clf_names):
        estimators.append(
            [
                f"{clf_name=}",
                get_classifier(
                    feature_node=features,
                    clf_name=clf_name,
                    task_name=task_name,
                    data_partition=data_partition,
                    train_test_split=train_test_split,
                ).model,
            ]
        )

    estimator = graph.instantiate_node(
        key=f"PrefittedVotingClassifier-{feat_name}".lower(),
        func=PrefittedVotingClassifier,
        kwargs=dict(estimators=estimators, voting="soft", verbose=10),
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


def basic_ensemble(
    dataset_name="anguita2013",
    modality="all",
    location="all",
    task_name="har",
    feat_name="ecdf",
    data_partition="predefined",
    fs_new=33,
    win_len=3,
    win_inc=1,
):
    dataset = dataset_importer(dataset_name)

    windowed_data = get_windowed_wearables(
        dataset=dataset, modality=modality, location=location, fs_new=fs_new, win_len=win_len, win_inc=win_inc
    )

    models = dict()

    train_test_splits = get_ancestral_metadata(windowed_data, "data_partitions")[data_partition]
    for train_test_split in randomised_order(train_test_splits):
        models[train_test_split] = ensemble_classifier(
            feat_name=feat_name,
            task_name=task_name,
            data_partition=data_partition,
            windowed_data=windowed_data,
            train_test_split=train_test_split,
            clf_names=["sgd", "lr", "rf"],
            evaluate=True,
        )

    return models


if __name__ == "__main__":
    basic_ensemble()
