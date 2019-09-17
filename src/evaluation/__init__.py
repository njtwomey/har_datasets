from .classification_metrics import *

__all__ = [
    'load_evaluation',
    'classification_metrics',
]


def load_evaluation(*args, **kwargs):
    evaluations = {kk: globals()[kk] for kk in __all__}
    assert args[0] in evaluations
    return evaluations[args[0]](*args[1:], **kwargs)
