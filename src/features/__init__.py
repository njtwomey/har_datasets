__all__ = [
    'FeatureBase', 'load_feature',
    'statistical_features'
]

from .base import *

from .statistical_features import statistical_features
from .ecdf_features import ecdf_features


def load_feature(name, *args, **kwargs):
    features = {kk: globals()[kk] for kk in __all__}
    assert name in features
    return features[name](*args, **kwargs)
