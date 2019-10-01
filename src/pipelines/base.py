from src.base import BaseGraph
from src.meta import PipelineMeta

__all__ = [
    'PipelineBase'
]


class PipelineBase(BaseGraph):
    def __init__(self, name, parent, *args, **kwargs):
        super(PipelineBase, self).__init__(
            name=name,
            parent=parent,
            meta=PipelineMeta(name)
        )
