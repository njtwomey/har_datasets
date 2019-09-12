from os.path import join

from .. import BaseGraph, FeatureMeta, PartitionAndWindow, PartitionAndWindowMetadata

__all__ = [
    'FeatureBase',
]


class FeatureBase(BaseGraph):
    def __init__(self, name, parent, *args, **kwargs):
        super(FeatureBase, self).__init__(
            name=name,
            parent=parent,
            meta=FeatureMeta(name),
        )
    
    @property
    def identifier(self):
        return join(
            self.parent.identifier,
            self.meta.name,
        )
