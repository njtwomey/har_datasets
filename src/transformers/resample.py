import numpy as np
from scipy import signal

from src.utils.decorators import PartitionByTrial

__all__ = [
    "resample",
]


def resample_data(index, data, fs_old, fs_new):
    if fs_old == fs_new:
        return data

    n_samples = int(data.shape[0] * fs_new / fs_old)

    return signal.resample(data, n_samples, axis=0)


def align_metadata(t_old, t_new):
    assert np.all((t_new[1:] - t_new[:-1]) > 0)
    assert np.all((t_old[1:] - t_old[:-1]) > 0)
    assert t_old[-1] == t_new[-1]
    assert t_old[0] == t_new[0]

    inds = [0]

    for ti, target in enumerate(t_new[1:], start=1):
        best, best_ind, last_ind = np.inf, None, inds[-1]
        for ii in range(last_ind, len(t_old)):
            diff_ii = abs(target - t_old[ii])
            is_best = diff_ii < best
            if is_best:
                best, best_ind = diff_ii, ii
            if (not is_best) or (ii == (len(t_old) - 1)):
                inds.append(best_ind)
                break

    inds = np.asarray(inds)

    return inds


def resample_metadata(index, data, fs_old, fs_new, is_index):
    if fs_old == fs_new:
        return data

    n_samples = int(data.shape[0] * fs_new / fs_old)
    t_old = index.time.values
    t_new = np.linspace(t_old[0], t_old[-1], n_samples)

    inds = align_metadata(t_old, t_new)
    df = data.iloc[inds].copy()
    if is_index:
        df.loc[:, "time"] = t_new
    df = df.astype(data.dtypes)
    return df


def resample(parent, fs_new):
    fs_old = parent.get_ancestral_metadata("fs")

    root = parent / f"{fs_new}Hz"
    root.meta.insert("fs", fs_new)

    kwargs = dict(fs_old=fs_old, fs_new=fs_new)

    if fs_old != fs_new:
        # Only compute indexes and outputs if the sample rate has changed
        for key, node in parent.index.items():
            root.index.create(
                key=key,
                func=PartitionByTrial(resample_metadata),
                kwargs=dict(index=parent.index["index"], data=node, is_index="index" in str(key), **kwargs),
            )

        for key, node in parent.outputs.items():
            root.outputs.create(
                key=key,
                func=PartitionByTrial(resample_data),
                kwargs=dict(index=parent.index["index"], data=node, **kwargs),
            )

    return root
