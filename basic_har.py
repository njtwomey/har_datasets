from src import models
from src.features import ecdf
from src.selectors import select_feats
from src.selectors import select_split
from src.selectors import select_task
from src.transformers import body_grav_filter
from src.transformers import resample
from src.transformers import window
from src.utils import accel_filt
from src.utils import gyro_filt
from src.utils import take_all
from src.utils.loaders import dataset_importer
from src.visualisations import umap_embedding


def main(
    dataset_name="anguita2013",
    fs_new=33,
    win_len=2.56,
    win_inc=1.0,
    task="har",
    split_type="predefined",
    feat_name="accel",
):
    source_filter = dict(all=take_all, accel=accel_filt, gyro=gyro_filt)[feat_name]
    dataset = dataset_importer(dataset_name)
    resampled = resample(parent=dataset, fs_new=fs_new)
    filtered = body_grav_filter(parent=resampled)
    windowed = window(parent=filtered, win_len=win_len, win_inc=win_inc)
    accel_feats = ecdf(parent=windowed, source_filter=source_filter, n_components=21)
    task = select_task(parent=accel_feats, task_name=task)
    split = select_split(parent=task, split_type=split_type)
    viz = umap_embedding(accel_feats, task=task)
    viz.evaluate_outputs()
    clf = models.sgd_classifier(parent=split, split=split, task=task, data=accel_feats)
    clf.evaluate_outputs()
    return clf


if __name__ == "__main__":
    main()
