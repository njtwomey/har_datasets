from src.base import BaseGraph

__all__ = ["VisualisationBase"]


class VisualisationBase(BaseGraph):
    def __init__(self, name, parent):
        super(VisualisationBase, self).__init__(
            name=name, parent=parent,
        )
