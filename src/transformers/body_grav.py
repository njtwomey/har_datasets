import pandas as pd
from numpy import isfinite

from scipy import signal

from .base import TransformerBase
from ..utils import transformer_decorator, Partition


def filter_signal(data, filter_order, cutoff, fs, btype, axis=0):
    ba = signal.butter(
        filter_order,
        cutoff / fs / 2,
        btype=btype
    )
    
    dd = signal.filtfilt(ba[0], ba[1], data, axis=axis)
    
    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(dd, columns=data.columns)
    
    return dd


@transformer_decorator
def body_filt(key, index, data, **kwargs):
    df = filter_signal(data, btype='high', **kwargs)
    assert isfinite(df.values).all()
    return df


@transformer_decorator
def grav_filt(key, index, data, **kwargs):
    df = filter_signal(data, btype='low', **kwargs)
    assert isfinite(df.values).all()
    return df


@transformer_decorator
def body_jerk_filt(key, index, data, **kwargs):
    df = body_filt(key, index, data, **kwargs)
    return df.diff().fillna(0)


def copy_file(key, index, data, **kwargs):
    return data


class body_grav(TransformerBase):
    def __init__(self, parent):
        super(body_grav, self).__init__(
            name=self.__class__.__name__,
            parent=parent,
        )
        
        kwargs = dict(filter_order=3, cutoff=0.3, fs=self.meta['fs'])
        for key, node in parent.index.items():
            self.index.clone_from_parent(parent=parent, key=key)
        
        for key, node in parent.outputs.items():
            self.outputs.add_output(
                key=key + ('body',),
                func=Partition(func=body_filt),
                sources=dict(data=node, index=parent.index.index),
                **kwargs,
            )
            
            self.outputs.add_output(
                key=key + ('body', 'jerk',),
                func=Partition(func=body_jerk_filt),
                sources=dict(data=node, index=parent.index.index),
                **kwargs,
            )
            
            if 'accel' in key:
                self.outputs.add_output(
                    key=key + ('grav',),
                    func=Partition(func=grav_filt),
                    sources=dict(data=node, index=parent.index.index),
                    **kwargs,
                )
