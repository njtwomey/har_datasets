from os.path import join

from numpy import concatenate

from src.base import BaseGraph, make_key
from src.utils.logger import get_logger

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
    def __init__(self, name, parent, source_filter, source_name, *args, **kwargs):
        super(FeatureBase, self).__init__(
            name=join(name, source_name), parent=parent,
        )
        
        assert source_filter is None or callable(source_filter)
        
        self.source_filter = source_filter
        self.source_name = source_name
        
        self.locations_set = set(self.get_ancestral_metadata('placements'))
        self.modality_set = set(self.get_ancestral_metadata('modalities'))
    
    def prepare_outputs(self, endpoints, key, func, **kwargs):
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
            key: tuple(str)
            func:
            feats: None, or list(tuple(str))
                This defines the types of outputs that the user can define to be outputted. If all features that
                arose from the accelerometer on the wrist should be aggregated, feats should be specified as:
                ```python
                feats = [('accel', 'wrist'),]
                ```
                Here 'accel' and 'wrist' are defined in modality and location metadata files, but in general the
                intersection of the key (a tuple of strings) and the elements of feats are compared. When the size of
                the intersection matches the length of the elements of feat, this element is added.
                
            **kwargs:

        Returns:

        """
        key = make_key(key)
        node = self.outputs.make_output(
            key=key, func=func, **kwargs
        )
        
        logger.info(f"Adding {node.name} to complete feature set")
        all_key = make_key('all')
        
        if self.source_filter(key):
            endpoints[all_key][node.name] = node
        
        # elif isinstance(self.source_filter, list):
        #     for feat in self.source_filter:
        #         feat = make_key(feat)
        #         if set(feat).issubset(set(key)):
        #             logger.info(f"Building feature {feat}/{node.name}")
        #             endpoints[feat][node.name] = node
        #
        # else:
        #     endpoints[all_key][node.name] = node
    
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
                backend='none',
                **node_dict
            )
