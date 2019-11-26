from collections import defaultdict

from src.features.base import FeatureBase
from src.features.statistical_features_impl import t_feat, f_feat

__all__ = [
    'statistical_features'
]


class statistical_features(FeatureBase):
    def __init__(self, parent, source_filter, source_name):
        super(statistical_features, self).__init__(
            name=self.__class__.__name__, parent=parent,
            source_filter=source_filter, source_name=source_name,
        )
        
        kwargs = dict(
            fs=self.get_ancestral_metadata('fs')
        )
        
        endpoints = defaultdict(dict)
        
        # There are two feature categories defined here:
        #   1. Time domain
        #   2. Frequency domain
        #
        # And these get mapped from transformed data from two sources:
        #   1. Acceleration
        #   2. Gyroscope
        #
        # Assuming these two sources have gone through some body/gravity transformations
        # (eg from src.transformations.body_grav_filt) there will actually be several
        # more sources, eg:
        #   1. accel-body
        #   2. accel-body-jerk
        #   3. accel-body-jerk
        #   4. accel-grav
        #   5. gyro-body
        #   6. gyro-body-jerk
        #   7. gyro-body-jerk
        #
        # With more data sources this list will grows quickly.
        #
        # The feature types (time and frequency domain) are mapped to the transformed
        # sources in a particular way. For example, the frequency domain features are
        # *not* calculated on the gravity data sources. The loop below iterates through
        # all of the outputs of the previous node in the graph, and the logic within
        # the loop manages the correct mapping of functions to sources.
        #
        # Consult with the dataset table (tables/datasets.md) and see anguita2013 for
        # details.
        
        index = self.parent.index['index']
        for key, node in self.parent.outputs.items():
            key_td = key + ('td',)
            key_fd = key + ('fd',)
            
            if 'accel' in key:
                self.prepare_outputs(
                    endpoints=endpoints,
                    key=key_td,
                    func=t_feat,
                    index=index,
                    data=node,
                    **kwargs
                )
                
                if 'grav' not in key:
                    self.prepare_outputs(
                        endpoints=endpoints,
                        key=key_fd,
                        func=f_feat,
                        index=index,
                        data=node,
                        **kwargs
                    )
            
            if 'gyro' in key:
                self.prepare_outputs(
                    endpoints=endpoints,
                    key=key_td,
                    func=f_feat,
                    index=index,
                    data=node,
                    **kwargs
                )
                
                self.prepare_outputs(
                    endpoints=endpoints,
                    key=key_fd,
                    func=f_feat,
                    index=index,
                    data=node,
                    **kwargs
                )
        
        self.assign_outputs(endpoints)
