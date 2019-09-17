from .. import BaseGraph, VisualisationMeta
from ..utils.backends import PNGBackend


class VisualisationBase(BaseGraph):
    def __init__(self, name, parent):
        super(VisualisationBase, self).__init__(
            name=name,
            parent=parent,
            meta=VisualisationMeta(name)
        )
        
        self.add_backend('png', PNGBackend(self.fs_root))
