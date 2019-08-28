__all__ = [
    'Dataset', 'load_dataset',
    'anguita2013',
    'pamap2',
    'uschad',
]

from .base import Dataset

from .anguita2013 import anguita2013
from .pamap2 import pamap2
from .uschad import uschad


def load_dataset(name, *args, **kwargs):
    datasets = {kk: globals()[kk] for kk in __all__}
    assert name in datasets
    return datasets[name](*args, **kwargs)
