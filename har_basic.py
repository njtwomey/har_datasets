from src.base import get_ancestral_metadata
from src.motifs.features import get_features
from src.motifs.features import get_windowed_wearables
from src.motifs.models import get_classifier
from src.utils.loaders import dataset_importer
from src.utils.misc import randomised_order
from src.visualisations.umap_embedding import umap_embedding


def basic_har(
    #
    # Dataset
    dataset_name="pamap2",
    #
    # Representation sources
    modality="all",
    location="all",
    #
    # Task/split
    task_name="har",
    data_partition="predefined",
    #
    # Windowification
    fs_new=33,
    win_len=3,
    win_inc=1,
    #
    # Features
    feat_name="ecdf",
    clf_name="rf",
    #
    # Embedding visualisation
    viz=False,
    evaluate=False,
):
    dataset = dataset_importer(dataset_name)

    # Resample, filter and window the raw sensor data
    wear_windowed = get_windowed_wearables(
        dataset=dataset, modality=modality, location=location, fs_new=fs_new, win_len=win_len, win_inc=win_inc
    )

    # Extract features
    features = get_features(feat_name=feat_name, windowed_data=wear_windowed)

    # Visualise the feature embeddings
    if viz:
        umap_embedding(features, task_name=task_name).evaluate()

    # Get classifier params
    models = dict()
    train_test_splits = get_ancestral_metadata(features, "data_partitions")[data_partition]
    for train_test_split in randomised_order(train_test_splits):
        models[train_test_split] = get_classifier(
            clf_name=clf_name,
            feature_node=features,
            task_name=task_name,
            data_partition=data_partition,
            evaluate=evaluate,
            train_test_split=train_test_split,
        )

    return features, models


if __name__ == "__main__":
    basic_har(viz=True, evaluate=True)
