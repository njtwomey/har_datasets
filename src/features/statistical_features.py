from .base import FeatureBase
from .statistical_features_impl import t_feat, f_feat


class statistical_features(FeatureBase):
    def __init__(self, parent):
        super(statistical_features, self).__init__(
            name=self.__class__.__name__,
            parent=parent,
        )
        
        kwargs = dict(
            fs=self.get_ancestral_metadata('fs')
        )
        
        for key, node in self.parent.index.items():
            self.index.clone_from_parent(parent=parent, key=key)
        
        index = self.parent.index['index']
        for key, node in self.parent.outputs.items():
            sources = dict(index=index, data=node, )
            
            if 'accel' in key:
                self.outputs.add_output(
                    key=key + ('td',),
                    func=t_feat,
                    sources=sources,
                    **kwargs
                )
                if 'grav' not in key:
                    self.outputs.add_output(
                        key=key + ('fd',),
                        func=f_feat,
                        sources=sources,
                        **kwargs
                    )
            if 'gyro' in key:
                self.outputs.add_output(
                    key=key + ('td',),
                    func=t_feat,
                    sources=sources,
                    **kwargs
                )
                self.outputs.add_output(
                    key=key + ('fd',),
                    func=f_feat,
                    sources=sources,
                    **kwargs
                )
