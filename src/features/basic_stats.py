from collections import defaultdict

from ..utils import feature_decorator

from .base import FeatureBase


def extract(index, data, *args, **kwargs):
    return


class basic_stats(FeatureBase):
    def __init__(self, dataset):
        super(basic_stats, self).__init__(
            feature=self.__class__.__name__,
            dataset=dataset,
            outputs=[
                ('accel', 'waist', 'jerk'),
                ('accel', 'waist', 'hpf'),
                ('accel', 'waist', 'lpf'),
            ]
        )
    
    @feature_decorator
    def build_features(self, index, data, modality, location, fs):
        raise NotImplementedError
