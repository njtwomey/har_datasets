from ..utils.backends import ScikitLearnBackend
from .. import BaseGraph, ModelMeta


class ModelBase(BaseGraph):
    def __init__(self, name, parent, model, *args, **kwargs):
        super(ModelBase, self).__init__(
            name=name,
            parent=parent,
            meta=ModelMeta(name)
        )
        
        self.model = model
        
        self.add_backend('sklearn', ScikitLearnBackend(self.fs_root))
