from src.base import BaseGraph
from src.meta import TransformerMeta

__all__ = [
    'TransformerBase'
]


class TransformerBase(BaseGraph):
    def __init__(self, name, parent, *args, **kwargs):
        super(TransformerBase, self).__init__(
            name=name,
            parent=parent,
            meta=TransformerMeta(name)
        )
