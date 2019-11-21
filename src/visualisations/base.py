from src.base import BaseGraph
from src.meta import VisualisationMeta

from mldb.backends import PNGBackend
from src.backends import ScikitLearnBackend

__all__ = [
    'VisualisationBase'
]


class VisualisationBase(BaseGraph):
    def __init__(self, name, parent):
        super(VisualisationBase, self).__init__(
            name=name,
            parent=parent,
            meta=VisualisationMeta(name)
        )
        
        self.add_backend('sklearn', ScikitLearnBackend(self.fs_root))
        self.add_backend('png', PNGBackend(self.fs_root))
