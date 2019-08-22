import pandas as pd
from scipy import signal

from .base import TransformerBase
from .. import FuncKeyMap


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


def body_filt(key, index, data, **kwargs):
    df = filter_signal(data, btype='high', **kwargs)
    return df


def grav_filt(key, index, data, **kwargs):
    df = filter_signal(data, btype='low', **kwargs)
    return df


def body_jerk_filt(key, index, data, **kwargs):
    df = body_filt(key, index, data, **kwargs)
    return df.diff().fillna(0)


class body_grav(TransformerBase):
    def __init__(self, parent):
        super(body_grav, self).__init__(
            name=self.__class__.__name__,
            parent=parent,
        )
        
        for key in parent.outputs.keys():
            self.add_outputs(FuncKeyMap(key, key + ('body',), body_filt))
            self.add_outputs(FuncKeyMap(key, key + ('body', 'jerk',), body_jerk_filt))
            if 'accel' in key:
                self.add_outputs(FuncKeyMap(key, key + ('grav',), grav_filt))
        
        self.add_extra_kwargs(
            filter_order=self.meta['filter_order'],
            cutoff=self.meta['cutoff'],
        )
