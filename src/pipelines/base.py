from src.base import BaseGraph

__all__ = ["PipelineBase"]


class PipelineBase(BaseGraph):
    def __init__(self, name, parent, *args, **kwargs):
        super(PipelineBase, self).__init__(
            name=name, parent=parent,
        )
