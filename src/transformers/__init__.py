__all__ = [
    'TransformerBase', 'load_transformer',
    'body_grav',
]

from .base import *

from .body_grav import body_grav


def load_transformer(name, *args, **kwargs):
    transformers = {kk: globals()[kk] for kk in __all__}
    assert name in transformers
    return transformers[name](*args, **kwargs)
