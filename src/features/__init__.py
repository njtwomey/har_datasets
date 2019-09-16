__all__ = [
    'FeatureBase', 'load_feature',
    'statistical_features', 'ecdf_11', 'ecdf_21'
]

from .base import *

from .statistical_features import statistical_features
from .ecdf_features import ecdf_11, ecdf_21


def load_feature(name, *args, **kwargs):
    features = {kk: globals()[kk] for kk in __all__}
    assert name in features
    return features[name](*args, **kwargs)
