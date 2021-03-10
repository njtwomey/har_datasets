from src.features import ecdf
from src.features import statistical_features
from src.models import random_forest
from src.models import sgd_classifier
from src.selectors import select_split
from src.selectors import select_task
from src.transformers import body_grav_filter
from src.transformers import resample
from src.transformers import window
from src.transformers.modality_selector import modality_selector
from src.utils.loaders import dataset_importer
from src.visualisations import umap_embedding


def har_basic(
    dataset_name="anguita2013",
    fs_new=33,
    win_len=2.56,
    win_inc=1.0,
    task="har",
    split_type="predefined",
    features="ecdf",
    modality="accel",
    placement="all",
    classifier="sgd",
):
    # Window/align the raw data
    dataset = dataset_importer(dataset_name)

    wear_resampled = resample(parent=dataset, fs_new=fs_new)
    wear_separated = body_grav_filter(parent=wear_resampled)
    wear_windowed = window(parent=wear_separated, win_len=win_len, win_inc=win_inc)

    # Calculate the features that we want
    if features == "statistical":
        wear_feats = statistical_features(parent=wear_windowed)
    elif features == "ecdf":
        wear_feats = ecdf(parent=wear_windowed, n_components=21)
    else:
        raise ValueError

    # Select the features that we're interested in, and get the features
    selected_feats = modality_selector(parent=wear_feats, modality=modality, placement=placement)

    # Get the task (and its labels), and the train/val/test splits
    task = select_task(parent=selected_feats, task_name=task)
    split = select_split(parent=task, split_type=split_type)

    # Learn the classifier
    if classifier == "sgd":
        clf = sgd_classifier(parent=split, split=split, task=task, data=selected_feats)
    elif classifier == "rf":
        clf = random_forest(parent=split, split=split, task=task, data=selected_feats)
    else:
        raise ValueError
    clf.evaluate_outputs()

    # Visualise the embeddings
    viz = umap_embedding(selected_feats, task=task)
    viz.evaluate_outputs()

    return selected_feats, task, split, clf


if __name__ == "__main__":
    har_basic()
