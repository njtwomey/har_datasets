import numpy as np

from .base import FeatureBase
from .. import feature_decorator

__all__ = [
    'ecdf_11',
]


def ecdf_rep(data, components):
    # From: https://github.com/nhammerla/ecdfRepresentation/blob/master/python/ecdfRep.py
    m = data.mean(0)
    data = np.sort(data, axis=0)
    data = data[np.int32(np.around(np.linspace(0, data.shape[0] - 1, num=components))), :]
    data = data.flatten(1)
    return np.hstack((data, m))


@feature_decorator
def calc_ecdf(key, index, data, n_components):
    return np.asarray([ecdf_rep(datum, n_components) for datum in data])


class ecdf_features(FeatureBase):
    def __init__(self, name, parent, n_components):
        super(ecdf_features, self).__init__(
            name=name, parent=parent,
        )
        
        self.index.clone_all_from_parent(parent=parent)
        
        for key, node in parent.outputs.items():
            self.outputs.add_output(
                key=key + ('ecdf',),
                func=calc_ecdf,
                sources=dict(index=parent.index['index'], data=node, ),
                n_components=n_components
            )


class ecdf_11(ecdf_features):
    def __init__(self, parent):
        super(ecdf_11, self).__init__(
            name=self.__class__.__name__,
            parent=parent,
            n_components=11,
        )
