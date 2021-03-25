from mldb import NodeWrapper

from src.features.ecdf_features import ecdf
from src.features.statistical_features import statistical_features
from src.transformers.body_grav_filter import body_grav_filter
from src.transformers.resample import resample
from src.transformers.source_selector import source_selector
from src.transformers.window import window


def get_windowed_wearables(
    dataset: NodeWrapper, modality: str, location: str, fs_new: float, win_len: float, win_inc: float
):
    selected_sources = source_selector(parent=dataset, modality=modality, location=location)
    wear_resampled = resample(parent=selected_sources, fs_new=fs_new)
    wear_filtered = body_grav_filter(parent=wear_resampled)
    wear_windowed = window(parent=wear_filtered, win_len=win_len, win_inc=win_inc)
    return wear_windowed


def get_features(feat_name: str, windowed_data: NodeWrapper):
    if feat_name == "statistical":
        features = statistical_features(parent=windowed_data)
    elif feat_name == "ecdf":
        features = ecdf(parent=windowed_data, n_components=21)
    else:
        raise ValueError

    assert isinstance(features, NodeWrapper)

    return features
