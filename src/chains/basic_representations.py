from src import dataset_importer, transformers, features

__all__ = [
    'stat_feat', 'ecdf_11', 'ecdf_21',
]


def dataset_at_33hz_2s56_1s0(name):
    dataset = dataset_importer(name)
    # resampled = transformers.resample_33(parent=dataset)
    filtered = transformers.body_grav_filter(parent=dataset)
    windowed = transformers.window_256_1(parent=filtered)
    return windowed


def stat_feat(name):
    windowed = dataset_at_33hz_2s56_1s0(name)
    feats = features.statistical_features(parent=windowed)
    return feats


def ecdf_11(name):
    windowed = dataset_at_33hz_2s56_1s0(name)
    feats = features.ecdf_11(parent=windowed)
    return feats


def ecdf_21(name):
    windowed = dataset_at_33hz_2s56_1s0(name)
    feats = features.ecdf_21(parent=windowed)
    return feats
