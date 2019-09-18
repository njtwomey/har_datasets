from .statistical_feature_repr import statistical_feature_repr

__all__ = [
    'load_representation',
    'statistical_feature_repr',
]


def load_representation(*args, **kwargs):
    representations = {kk: globals()[kk] for kk in __all__}
    assert args[0] in representations
    return representations[args[0]](*args[1:], **kwargs)
