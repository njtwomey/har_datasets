from src.base import BaseGraph

__all__ = ["TransformerBase"]


class TransformerBase(BaseGraph):
    def __init__(self, name, parent, *args, **kwargs):
        super(TransformerBase, self).__init__(
            name=name, parent=parent,
        )
