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

    def filter_outputs(self, key):
        self.evaluate_outputs()
        return dict(map(
            lambda key: (key, self.outputs[key].evaluate()), filter(lambda kk: key in kk, map(str, self.outputs.keys()))
        ))

    def get_results(self):
        return self.filter_outputs('results')

    @property
    def deployed(self):
        assert self.parent.name == 'deployable'
        return self.models[0].model
