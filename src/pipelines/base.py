from .. import BaseGraph, PipelineMeta


class TransformerBase(BaseGraph):
    def __init__(self, name, parent, *args, **kwargs):
        super(TransformerBase, self).__init__(
            name=name,
            parent=parent,
            meta=PipelineMeta(name)
        )
