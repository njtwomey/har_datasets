from src import dot_env_decorator

from src.utils import take_all

from src import dataset_importer
from src import features
from src import transformers
from src import selectors
from src import models


@dot_env_decorator
def main(dataset_name='anguita2013', fs_new=33, win_len=2.56, win_inc=1.0, task='har', split_type='predefined',
         feat_name='all'):
    dataset = dataset_importer(dataset_name)
    resampled = transformers.resample(parent=dataset, fs_new=fs_new)
    filtered = transformers.body_grav_filter(parent=resampled)
    windowed = transformers.window(parent=filtered, win_len=win_len, win_inc=win_inc)
    accel_feats = features.ecdf(parent=windowed, source_filter=take_all(), source_name=feat_name, n_components=21)
    task = selectors.select_task(parent=accel_feats, task_name=task)
    split = selectors.select_split(parent=task, split_type=split_type)
    clf = models.sgd_classifier(parent=split, split=split, task=task, data=accel_feats)
    clf.evaluate_outputs()
    return clf


if __name__ == '__main__':
    main()
