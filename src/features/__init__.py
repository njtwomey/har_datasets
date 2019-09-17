from .base import *

from .statistical_features import *
from .ecdf_features import *


def load_feature(name, *args, **kwargs):
    features = {kk: globals()[kk] for kk in __all__}
    assert name in features
    return features[name](*args, **kwargs)
