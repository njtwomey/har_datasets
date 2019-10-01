from src.backends import ScikitLearnBackend
from src.base import BaseGraph
from src.meta import ModelMeta

__all__ = [
    "ModelBase"
]


class ModelBase(BaseGraph):
    def __init__(self, name, parent, model, *args, **kwargs):
        super(ModelBase, self).__init__(
            name=name,
            parent=parent,
            meta=ModelMeta(name)
        )
        
        self.model = model
        
        self.add_backend('sklearn', ScikitLearnBackend(self.fs_root))
