import pandas as pd
from numpy import atleast_2d, atleast_3d

from .base import TransformerBase, Partition

from .. import transformer_decorator
from .. import sliding_window_rect


@transformer_decorator
def window_data(key, index, data, fs, win_len, win_inc):
    win_len = int(win_len * fs)
    win_inc = int(win_inc * fs)
    data_windowed = sliding_window_rect(
        atleast_2d(data.values), win_len, win_inc
    )
    return atleast_3d(data_windowed)


def window_index(key, index, data, fs, win_len, win_inc):
    assert isinstance(data, pd.DataFrame)
    data_windowed = window_data(key, index, data, fs, win_len, win_inc)
    return pd.DataFrame(data_windowed[:, 0, :], columns=data.columns, )


class window(TransformerBase):
    def __init__(self, name, parent, win_len, win_inc, fs):
        super(window, self).__init__(
            name=name, parent=parent,
        )
        
        kwargs = dict(fs=fs, win_len=win_len, win_inc=win_inc)
        
        # Build index outputs
        for key, node in parent.index.items():
            self.index.add_output(
                key=key,
                func=Partition(window_index),
                sources=dict(
                    index=parent.index['index'],
                    data=node,
                ),
                **kwargs
            )
        
        # Build Data outputs
        for key, node in parent.outputs.items():
            self.outputs.add_output(
                key=key,
                func=Partition(window_data),
                sources=dict(
                    index=parent.index['index'],
                    data=node,
                ),
                **kwargs
            )


class window_128_1(window):
    def __init__(self, parent):
        super(window_128_1, self).__init__(
            name=self.__class__.__name__,
            parent=parent,
            win_len=1.28,
            win_inc=1.0,
            fs=parent.meta['fs']
        )


class window_256_1(window):
    def __init__(self, parent):
        super(window_256_1, self).__init__(
            name=self.__class__.__name__,
            parent=parent,
            win_len=2.56,
            win_inc=1.0,
            fs=parent.meta['fs']
        )
