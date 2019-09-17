from collections import defaultdict
from numpy import concatenate

from .base import FeatureBase
from .statistical_features_impl import t_feat, f_feat

__all__ = [
    'statistical_features'
]


def concatenate_sources(key, **datas):
    assert len(set([type(i) for i in datas.values()])) == 1
    return concatenate([
        datas[kk] for kk in sorted(datas.keys())
    ], axis=1)


class statistical_features(FeatureBase):
    def __init__(self, parent):
        super(statistical_features, self).__init__(
            name=self.__class__.__name__,
            parent=parent,
        )
        
        kwargs = dict(
            fs=self.get_ancestral_metadata('fs')
        )
        
        endpoints = defaultdict(dict)
        feats = ('td', 'fd')
        
        index = self.parent.index['index']
        for key, node in self.parent.outputs.items():
            sources = dict(index=index, data=node)
            
            key_td = key + ('td',)
            key_fd = key + ('fd',)
            
            if 'accel' in key:
                self.pre_aggregate_output(
                    endpoints=endpoints,
                    key=key_td,
                    func=t_feat,
                    sources=sources,
                    feats=feats,
                    **kwargs
                )
                
                if 'grav' not in key:
                    self.pre_aggregate_output(
                        endpoints=endpoints,
                        key=key_fd,
                        func=f_feat,
                        sources=sources,
                        feats=feats,
                        **kwargs
                    )
            
            if 'gyro' in key:
                self.pre_aggregate_output(
                    endpoints=endpoints,
                    key=key_td,
                    func=f_feat,
                    sources=sources,
                    feats=feats,
                    **kwargs
                )
                
                self.pre_aggregate_output(
                    endpoints=endpoints,
                    key=key_fd,
                    func=f_feat,
                    sources=sources,
                    feats=feats,
                    **kwargs
                )
        
        self.aggregate_outputs(endpoints)
