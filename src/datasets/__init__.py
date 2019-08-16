__all__ = [
    'anguita2013',
    'pamap2',
    'uschad'
]

from .anguita2013 import anguita2013
from .pamap2 import pamap2
from .uschad import uschad

processors = {kk: globals()[kk] for kk in __all__}
