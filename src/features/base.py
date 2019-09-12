from os.path import join

from .. import BaseGraph, FeatureMeta, PartitionAndWindow, PartitionAndWindowMetadata

__all__ = [
    'FeatureBase',
]


class FeatureBase(BaseGraph):
    def __init__(self, name, parent, *args, **kwargs):
        super(FeatureBase, self).__init__(name=name, parent=parent)
        
        self.meta = FeatureMeta(name)
        
    @property
    def identifier(self):
        return join(
            self.parent.identifier,
            self.meta.name,
        )
    
    def compose_meta(self, key):
        return self.node(
            node_name=self.build_path(key),
            func=PartitionAndWindowMetadata(
                win_len=self.meta.window.win_len,
                win_inc=self.meta.window.win_inc,
            ),
            sources=dict(
                index=self.parent.index,
                data=self.parent.nodes[
                    self.parent.build_path(key)
                ]
            ),
            kwargs=dict(
                key=key,
                **(self.extra_args or dict())
            )
        )
    
    def make_node(self, in_key, out_key, func):
        node = self.node(
            node_name=self.build_path(*out_key),
            func=PartitionAndWindow(
                win_len=self.meta.window.win_len,
                win_inc=self.meta.window.win_inc,
                func=func,
            ),
            sources=dict(
                index=self.parent.index,
                data=self.parent.outputs[in_key]
            ),
            kwargs=dict(
                key=out_key,
                **(self.extra_args or dict())
            )
        )
        
        return node
