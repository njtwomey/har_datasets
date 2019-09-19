from os.path import join
from collections import defaultdict

import numpy as np

from .base import FeatureBase
from .. import feature_decorator

__all__ = [
    'ecdf',
]


def ecdf_rep(data, components):
    # Taken from: https://github.com/nhammerla/ecdfRepresentation/blob/master/python/ecdfRep.py
    m = data.mean(0)
    data = np.sort(data, axis=0)
    data = data[np.int32(np.around(np.linspace(0, data.shape[0] - 1, num=components))), :]
    data = data.flatten(1)
    return np.hstack((data, m))


@feature_decorator
def calc_ecdf(key, index, data, n_components):
    return np.asarray([ecdf_rep(datum, n_components) for datum in data])


class ecdf(FeatureBase):
    def __init__(self, parent, n_components):
        super(ecdf, self).__init__(
            name=self.__class__.__name__, parent=parent,
        )
        
        endpoints = defaultdict(dict)
        
        self.n_components = n_components
        
        for key, node in parent.outputs.items():
            self.pre_aggregate_output(
                endpoints=endpoints,
                key=key + ('ecdf',),
                func=calc_ecdf,
                sources=dict(index=parent.index['index'], data=node, ),
                n_components=n_components
            )
        
        self.aggregate_outputs(endpoints)
    
    @property
    def identifier(self):
        return join(
            self.parent.identifier,
            f'{self.name}_{self.n_components}',
        )
