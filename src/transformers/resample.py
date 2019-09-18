import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors

from scipy import signal
from scipy.interpolate import interp1d

from .base import TransformerBase
from .. import Partition

__all__ = [
    'resample_33',
]


def resample_data(key, index, data, fs_old, fs_new):
    n_samples = int(data.shape[0] * fs_new / fs_old)
    return signal.resample(data, n_samples, axis=0)


def resample_metadata(key, index, data, fs_old, fs_new):
    n_samples = int(data.shape[0] * fs_new / fs_old)
    t_old = index.time.values
    t_new = np.linspace(t_old[0], t_old[-1], n_samples)
    
    assert t_old[0] == t_new[0]
    assert t_old[-1] == t_new[-1]
    
    # Since metadata is discrete, we take a nearest-neighbour approach to resample.
    knn1 = NearestNeighbors(1)
    knn1.fit(t_old[:, None])
    _, inds = knn1.kneighbors(t_new[:, None], 1)
    inds = inds.ravel()

    df = data.iloc[inds]
    
    return df


class resampler(TransformerBase):
    def __init__(self, name, parent, fs_new):
        super(resampler, self).__init__(
            name=name, parent=parent
        )
        
        kwargs = dict(
            fs_old=parent.get_ancestral_metadata('fs'),
            fs_new=fs_new
        )
        
        for key, node in parent.index.items():
            self.index.add_output(
                key=key,
                func=Partition(resample_metadata),
                sources=dict(
                    index=parent.index['index'],
                    data=node,
                ),
                **kwargs
            )
        
        for key, node in parent.outputs.items():
            self.outputs.add_output(
                key=key,
                func=Partition(resample_data),
                sources=dict(
                    index=parent.index['index'],
                    data=node,
                ),
                **kwargs
            )


class resample_33(resampler):
    def __init__(self, parent):
        super(resample_33, self).__init__(
            name=self.__class__.__name__,
            parent=parent,
            fs_new=33
        )
