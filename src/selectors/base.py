from loguru import logger

from src.base import BaseGraph


__all__ = [
    "SelectorBase",
]


class SelectorBase(BaseGraph):
    def __init__(self, name, parent, *args, **kwargs):
        super(SelectorBase, self).__init__(name=name, parent=parent, meta=kwargs.get("meta", None))
