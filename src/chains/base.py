from .. import BaseGraph, ChainMeta


class ChainBase(BaseGraph):
    def __init__(self, name, parent, *args, **kwargs):
        super(ChainBase, self).__init__(
            name=name,
            parent=parent,
            meta=ChainMeta(name)
        )
