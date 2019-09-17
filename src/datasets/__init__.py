__all__ = [
    'Dataset', 'load_dataset',
    'anguita2013',
    'pamap2',
    'uschad',
]

from .base import *

from .anguita2013 import *
from .pamap2 import *
from .uschad import *


def load_dataset(name, *args, **kwargs):
    datasets = {kk: globals()[kk] for kk in __all__}
    assert name in datasets
    return datasets[name](*args, **kwargs)
