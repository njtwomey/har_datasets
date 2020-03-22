from src.base import BaseGraph

__all__ = [
    "ModelBase"
]


class ModelBase(BaseGraph):
    def __init__(self, name, parent, model, *args, **kwargs):
        super(ModelBase, self).__init__(
            name=name,
            parent=parent,
        )
        
        self.models = {}

    # @property
    # def model(self):
    #     return self.outputs['model']
    #
    # @property
    # def results(self):
    #     return self.outputs['results']
    #
    # @property
    # def preds(self):
    #     return self.outputs['preds']
    #
    # @property
    # def probs(self):
    #     return self.outputs['probs']
    #
    # @property
    # def deployed(self):
    #     assert self.parent.name == 'deployable'
    #     return self.models[0].model
