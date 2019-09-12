__all__ = [
    'TransformerBase', 'load_transformer',
    'body_grav', 'window_256_1', 'resample_33'
]

from .base import *

from .body_grav import body_grav
from .window import window_256_1
from .resample import resample_33


def load_transformer(name, *args, **kwargs):
    transformers = {kk: globals()[kk] for kk in __all__}
    assert name in transformers, f'Transformer "{name}" cannot be found in list of transformations'
    return transformers[name](*args, **kwargs)
