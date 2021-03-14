from src.features import ecdf
from src.features import statistical_features
from src.models import random_forest
from src.models import sgd_classifier
from src.selectors import select_split
from src.selectors import select_task
from src.transformers import body_grav_filter
from src.transformers import resample
from src.transformers import window
from src.transformers.modality_selector import concatenate_features
from src.transformers.modality_selector import modality_selector
from src.utils.loaders import dataset_importer
from src.visualisations import umap_embedding


def har_basic(
    dataset_name="anguita2013",
    fs_new=33,
    win_len=3,
    win_inc=1,
    task="har",
    split_type="predefined",
    features="ecdf",
    modality="accel",
    location="all",
    classifier="sgd",
    viz=False,
):
    # Window/align the raw data
    dataset = dataset_importer(dataset_name)

    # Select the features that we're interested in, and get the features
    selected_feats = modality_selector(parent=dataset, modality=modality, location=location)

    # Process the wearable data
    wear_resampled = resample(parent=selected_feats, fs_new=fs_new)
    wear_filtered = body_grav_filter(parent=wear_resampled)
    wear_windowed = window(parent=wear_filtered, win_len=win_len, win_inc=win_inc)

    # Calculate the features that we want
    if features == "statistical":
        wear_feats = statistical_features(parent=wear_windowed)
    elif features == "ecdf":
        wear_feats = ecdf(parent=wear_windowed, n_components=21)
    else:
        raise ValueError

    # Aggregate the features together
    feature_node = concatenate_features(wear_feats)

    # Get the task (and its labels), and the train/val/test splits
    task = select_task(parent=wear_feats, task_name=task)
    split = select_split(parent=task, split_type=split_type)

    # Learn the classifier
    if classifier == "sgd":
        clf = sgd_classifier(parent=split, data=feature_node)
    elif classifier == "rf":
        clf = random_forest(parent=split, data=feature_node)
    else:
        raise ValueError

    clf.dump_graph()
    clf.evaluate()

    # Visualise the embeddings
    if viz:
        umap_embedding(feature_node, task=task).evaluate()

    return feature_node, task, split, clf


if __name__ == "__main__":
    har_basic()
