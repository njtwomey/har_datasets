import numpy as np
from scipy import signal

from src.utils.decorators import PartitionByTrial

__all__ = [
    "body_grav_filter",
]


def filter_signal(data, filter_order, cutoff, fs, btype, axis=0):
    ba = signal.butter(filter_order, cutoff / fs / 2, btype=btype)

    mu = np.mean(data, axis=0, keepdims=True)

    dd = signal.filtfilt(ba[0], ba[1], data - mu, axis=axis) + mu

    return dd


def body_filt(index, data, **kwargs):
    filt = filter_signal(data, btype="high", **kwargs)
    assert np.isfinite(filt).all()
    return filt


def grav_filt(index, data, **kwargs):
    filt = filter_signal(data, btype="low", **kwargs)
    assert np.isfinite(filt).all()
    return filt


def body_jerk_filt(index, data, **kwargs):
    filt = body_filt(index, data, **kwargs)
    jerk = np.empty(filt.shape, dtype=filt.dtype)
    jerk[0] = 0
    jerk[1:] = filt[1:] - filt[:-1]
    assert np.isfinite(filt).all()
    return jerk


def body_grav_filter(parent):
    root = parent / "body_grav_filter"

    kwargs = dict(fs=root.get_ancestral_metadata("fs"), filter_order=3, cutoff=0.3)

    for key, node in parent.outputs.items():
        root.outputs.add_output(
            key=key + ("body",),
            func=PartitionByTrial(func=body_filt),
            backend="none",
            kwargs=dict(data=node, index=parent.index.index, **kwargs),
        )

        root.outputs.add_output(
            key=key + ("body", "jerk"),
            func=PartitionByTrial(func=body_jerk_filt),
            backend="none",
            kwargs=dict(data=node, index=parent.index.index, **kwargs),
        )

        if "accel" in key:
            root.outputs.add_output(
                key=key + ("grav",),
                func=PartitionByTrial(func=grav_filt),
                backend="none",
                kwargs=dict(data=node, index=parent.index.index, **kwargs),
            )

    return root
