import numpy as np
from scipy import signal

from .base import TransformerBase
from ..utils import transformer_decorator, Partition

__all__ = [
    'body_grav_filter',
]


def filter_signal(data, filter_order, cutoff, fs, btype, axis=0):
    ba = signal.butter(
        filter_order,
        cutoff / fs / 2,
        btype=btype
    )
    
    dd = signal.filtfilt(ba[0], ba[1], data, axis=axis)
    
    return dd


@transformer_decorator
def body_filt(key, index, data, **kwargs):
    filt = filter_signal(data, btype='high', **kwargs)
    assert np.isfinite(filt).all()
    return filt


@transformer_decorator
def grav_filt(key, index, data, **kwargs):
    filt = filter_signal(data, btype='low', **kwargs)
    assert np.isfinite(filt).all()
    return filt


@transformer_decorator
def body_jerk_filt(key, index, data, **kwargs):
    filt = body_filt(key, index, data, **kwargs)
    jerk = np.empty(filt.shape, dtype=filt.dtype)
    jerk[0] = 0
    jerk[1:] = filt[1:] - filt[:-1]
    assert np.isfinite(filt).all()
    return jerk


class body_grav_filter(TransformerBase):
    def __init__(self, parent):
        super(body_grav_filter, self).__init__(
            name=self.__class__.__name__,
            parent=parent,
        )
        
        kwargs = dict(
            filter_order=3, cutoff=0.3,
            fs=self.get_ancestral_metadata('fs')
        )
        
        for key, node in parent.outputs.items():
            self.outputs.add_output(
                key=key + ('body',),
                func=Partition(func=body_filt),
                sources=dict(data=node, index=parent.index.index),
                backend='none',
                **kwargs,
            )
            
            self.outputs.add_output(
                key=key + ('body', 'jerk',),
                func=Partition(func=body_jerk_filt),
                sources=dict(data=node, index=parent.index.index),
                backend='none',
                **kwargs,
            )
            
            if 'accel' in key:
                self.outputs.add_output(
                    key=key + ('grav',),
                    func=Partition(func=grav_filt),
                    sources=dict(data=node, index=parent.index.index),
                    backend='none',
                    **kwargs,
                )
