from numpy import concatenate

from .. import BaseGraph, FeatureMeta

__all__ = [
    'FeatureBase',
]


def concatenate_sources(key, **datas):
    assert len(set([type(i) for i in datas.values()])) == 1
    return concatenate([
        datas[kk] for kk in sorted(datas.keys())
    ], axis=1)


class FeatureBase(BaseGraph):
    def __init__(self, name, parent, *args, **kwargs):
        super(FeatureBase, self).__init__(
            name=name,
            parent=parent,
            meta=FeatureMeta(name),
        )
    
    def pre_aggregate_output(self, endpoints, key, func, sources, feats=None, **kwargs):
        node = self.outputs.make_output(
            key=key, func=func, sources=sources, **kwargs
        )
        
        modality = next(filter(
            lambda k: k in set(self.get_ancestral_metadata('locations')), key
        ))
        
        location = next(filter(
            lambda k: k in set(self.get_ancestral_metadata('modalities')), key
        ))
        
        endpoints[('all',)][node.name] = node
        endpoints[(location, modality,)][node.name] = node
        for feat in feats or []:
            if feat in key:
                endpoints[(location, modality, feat,)][node.name] = node
    
    def aggregate_outputs(self, endpoints):
        for key, node_dict in endpoints.items():
            self.outputs.add_output(
                key=key,
                func=concatenate_sources,
                sources=node_dict,
                backend='none',
            )
