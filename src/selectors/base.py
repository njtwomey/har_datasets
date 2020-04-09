from src.base import BaseGraph
from src.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "SelectorBase",
]


class SelectorBase(BaseGraph):
    def __init__(self, name, parent, *args, **kwargs):
        super(SelectorBase, self).__init__(name=name, parent=parent, meta=kwargs.get("meta", None))
