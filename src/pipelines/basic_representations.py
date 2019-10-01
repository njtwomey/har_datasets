from src import dataset_importer
from src import features
from src import transformers

__all__ = [
    'stat_feat', 'ecdf_11', 'ecdf_21',
]


def dataset_windowed(name, fs_new=33, win_len=2.56, win_inc=1.0):
    dataset = dataset_importer(name)
    resampled = transformers.resample(parent=dataset, fs_new=fs_new)
    filtered = transformers.body_grav_filter(parent=resampled)
    windowed = transformers.window(parent=filtered, win_len=win_len, win_inc=win_inc)
    return windowed


def stat_feat(name):
    windowed = dataset_windowed(name)
    feats = features.statistical_features(parent=windowed)
    return feats


def ecdf(name, n_components):
    windowed = dataset_windowed(name)
    feats = features.ecdf(parent=windowed, n_components=n_components)
    return feats


def ecdf_11(name):
    return ecdf(name, 11)


def ecdf_21(name):
    return ecdf(name, 21)
