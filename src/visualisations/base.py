from .. import BaseGraph, VisualisationMeta
from ..utils.backends import ScikitLearnBackend, PNGBackend


class VisualisationBase(BaseGraph):
    def __init__(self, name, parent):
        super(VisualisationBase, self).__init__(
            name=name,
            parent=parent,
            meta=VisualisationMeta(name)
        )
        
        self.add_backend('sklearn', ScikitLearnBackend(self.fs_root))
        self.add_backend('png', PNGBackend(self.fs_root))
