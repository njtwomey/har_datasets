__all__ = [
    'Dataset',
    'anguita2013',
    'pamap2',
    'uschad',
]

from .base import Dataset

from .anguita2013 import anguita2013
from .pamap2 import pamap2
from .uschad import uschad

processors = {kk: globals()[kk] for kk in __all__}
