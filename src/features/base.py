from src.base import BaseGraph


__all__ = [
    "FeatureBase",
]


class FeatureBase(BaseGraph):
    def __init__(self, name, parent, *args, **kwargs):
        super(FeatureBase, self).__init__(name=name, parent=parent)

    @property
    def features(self):
        return self.outputs["features"]
