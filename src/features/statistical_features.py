from .base import FeatureBase
from .statistical_features_impl import t_feat, f_feat


class statistical_features(FeatureBase):
    def __init__(self, parent):
        super(statistical_features, self).__init__(
            name=self.__class__.__name__,
            parent=parent,
        )
        
        for key in self.parent.outputs.keys():
            if 'accel' in key:
                self.add_output(key, ('td',), t_feat)
                if 'grav' not in key:
                    self.add_output(key, ('fd',), f_feat)
            if 'gyro' in key:
                self.add_output(key, ('td',), t_feat)
                self.add_output(key, ('fd',), f_feat)
