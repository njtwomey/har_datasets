from os.path import join

import numpy as np
import pandas as pd
from loguru import logger

from src.transformers.base import TransformerBase
from src.utils.decorators import PartitionByTrial


__all__ = [
    "window",
]


def window_data(key, index, data, fs, win_len, win_inc):
    assert data.ndim == 2

    win_len = int(win_len * fs)
    win_inc = int(win_inc * fs)

    data_windowed = sliding_window_rect(data, win_len, win_inc)

    if data.shape[0] // win_len == 0:
        raise ValueError
    elif data.shape[0] // win_len == 1:
        return data_windowed[None, ...]
    elif data_windowed.ndim == 2:
        return data_windowed[..., None]
    assert data_windowed.ndim == 3
    return data_windowed


def window_index(key, index, data, fs, win_len, win_inc, slice_at="middle"):
    assert isinstance(data, pd.DataFrame)
    data_windowed = window_data(key, index, data.values, fs, win_len, win_inc)
    ind = dict(start=0, middle=data_windowed.shape[1] // 2, end=-1,)[slice_at]
    df = pd.DataFrame(data_windowed[:, ind, :], columns=data.columns)
    df = df.astype(data.dtypes)
    return df


class window(TransformerBase):
    def __init__(self, parent, win_len, win_inc):
        super(window, self).__init__(
            name=self.__class__.__name__, parent=parent,
        )

        self.win_len = win_len
        self.win_inc = win_inc

        fs = self.get_ancestral_metadata("fs")

        kwargs = dict(win_len=self.win_len, win_inc=self.win_inc, fs=fs,)

        # Build index outputs
        for key, node in parent.index.items():
            self.index.add_output(
                key=key,
                func=PartitionByTrial(window_index),
                kwargs=dict(index=parent.index["index"], data=node, **kwargs,),
            )

        # Build Data outputs
        for key, node in parent.outputs.items():
            self.outputs.add_output(
                key=key,
                func=PartitionByTrial(window_data),
                backend="none",
                kwargs=dict(index=parent.index["index"], data=node, **kwargs,),
            )

    @property
    def identifier(self):
        win_len = f"{self.win_len:03.2f}"
        win_inc = f"{self.win_inc:03.2f}"
        return self.parent.identifier / f"{self.name}_{win_len}s_{win_inc}s"


def norm_shape(shape):
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass

    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass

    logger.exception(TypeError("shape must be an int, or a tuple of ints"))


def sliding_window(a, ws, ss=None, flatten=True):
    """
    based on: https://stackoverflow.com/questions/22685274

    Return a sliding window over a in any number of dimensions

    Parameters
    ----------
    a : ndarray
        an n-dimensional numpy array
    ws : int, tuple
        an int (a is 1D) or tuple (a is 2D or greater) representing the size of
        each dimension of the window
    ss : int, tuple
        an int (a is 1D) or tuple (a is 2D or greater) representing the amount
        to slide the window in each dimension. If not specified, it defaults to ws.
    flatten : book
        if True, all slices are flattened, otherwise, there is an extra dimension
        for each dimension of the input.

    Returns
    -------
        strided : ndarray
            an array containing each n-dimensional window from a
    """

    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)

    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape), len(ws), len(ss)]
    if 1 != len(set(ls)):
        logger.exception(
            ValueError(f"a.shape, ws and ss must all have the same length. They were {ls}")
        )

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        logger.exception(
            ValueError(
                f"ws cannot be larger than a in any dimension. a.shape was %s and "
                "ws was {(str(a.shape), str(ws))}"
            )
        )

    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = np.lib.stride_tricks.as_strided(a, shape=newshape, strides=newstrides)
    if not flatten:
        return strided

    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    dim = list(filter(lambda i: i != 1, dim))

    return strided.reshape(dim)


def sliding_window_rect(data, length, increment):
    length = (length, data.shape[1])
    increment = (increment, data.shape[1])

    return sliding_window(data, length, increment)
