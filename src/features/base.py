from numpy import concatenate

from src.base import BaseGraph, make_key
from src.meta import FeatureMeta
from src.utils.logging import get_logger

logger = get_logger(__name__)

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
        
        self.locations_set = set(self.get_ancestral_metadata('locations').keys())
        self.modality_set = set(self.get_ancestral_metadata('modalities'))
    
    def prepare_outputs(self, endpoints, key, func, sources, feats=None, **kwargs):
        """
        Since the feature extraction function is applied to all of the input sources individually and it is not
        (necessarily) desirable to perform classification analysis on features from each stream individually, this
        function allows the features to be extracted, but it does not append these features as outputs. Instead it
        a dictionary of nodes is built up and assigned to the variable endpoints.
        
        By default this function prepares for aggregating all features into one dictionary which is set as the sole
        output when the 'self.aggregate_outputs(endpoints)' is called. If other endpoints are desired, these can be
        specified with the feats variable, as described in its docstring.
        
        Args:
            endpoints: defaultdict(dict)
                This
            key:
            func:
            sources:
            feats: None, or list(tuple(str))
                This defines the types of outputs that the user can define to be outputted. If all features that
                arose from the accelerometer on the wrist should be aggregated, feats should be specified as:
                ```python
                feats = [('accel', 'wrist'),]
                ```
                where 'accel' and 'wrist' are defined in modality and location metadata files. Only when all items of
                the feats tuple are found in the 'key' argument to this function is the
            **kwargs:

        Returns:

        """
        node = self.outputs.make_output(
            key=key, func=func, sources=sources, **kwargs
        )
        
        endpoints[make_key('all')][node.name] = node
        for feat in feats or []:
            feat = make_key(feat)
            if len(set(feat) & set(key)) == len(feat):
                endpoints[feat][node.name] = node
    
    def assign_outputs(self, endpoints):
        """
        
        Args:
            endpoints:

        Returns:

        """
        for key, node_dict in endpoints.items():
            logger.info(f'Aggregates for feature {key}: {{{node_dict.keys()}}}')
            self.outputs.add_output(
                key=key,
                func=concatenate_sources,
                sources=node_dict,
                backend='none',
            )
