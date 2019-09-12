import numpy as np

from .base import FeatureBase
from .. import feature_decorator


def ecdf_rep(data, components):
    # From: https://github.com/nhammerla/ecdfRepresentation/blob/master/python/ecdfRep.py
    #
    #   rep = ecdfRep(data, components)
    #
    #   Estimate ecdf-representation according to
    #     Hammerla, Nils Y., et al. "On preserving statistical characteristics of
    #     accelerometry data using their empirical cumulative distribution."
    #     ISWC. ACM, 2013.
    #
    #   Input:
    #       data        Nxd     Input data (rows = samples).
    #       components  int     Number of components to extract per axis.
    #
    #   Output:
    #       rep         Mx1     Data representation with M = d*components+d
    #                           elements.
    #
    #   Nils Hammerla '15
    #
    m = data.mean(0)
    data = np.sort(data, axis=0)
    data = data[np.int32(np.around(np.linspace(0, data.shape[0] - 1, num=components))), :]
    data = data.flatten(1)
    return np.hstack((data, m))


@feature_decorator
def calc_ecdf(key, index, data, fs, n_components):
    return ecdf_rep(data.values, n_components)


class ecdf_features(FeatureBase):
    def __init__(self, parent):
        super(ecdf_features, self).__init__(
            name=self.__class__.__name__,
            parent=parent,
        )
        
        for key in self.parent.outputs.keys():
            self.add_output(key, 'ecdf', calc_ecdf)

        self.add_extra_kwargs(
            n_components=self.meta['n_components'],
        )
