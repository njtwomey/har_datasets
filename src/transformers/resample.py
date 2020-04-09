import numpy as np

from scipy import signal

from src.transformers.base import TransformerBase
from src.utils.decorators import PartitionByTrial

__all__ = [
    "resample",
]


def resample_data(key, index, data, fs_old, fs_new):
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


def resample_metadata(key, index, data, fs_old, fs_new):
    if fs_old == fs_new:
        return data

    n_samples = int(data.shape[0] * fs_new / fs_old)
    t_old = index.time.values
    t_new = np.linspace(t_old[0], t_old[-1], n_samples)

    inds = align_metadata(t_old, t_new)
    df = data.iloc[inds].copy()
    if "index" in key:
        df.loc[:, "time"] = t_new
    df = df.astype(data.dtypes)
    return df


class resample(TransformerBase):
    def __init__(self, parent: object, fs_new: object = None) -> object:
        super(resample, self).__init__(name=self.__class__.__name__, parent=parent)

        self.fs_old = parent.get_ancestral_metadata("fs")
        self.fs_new = fs_new
        if self.fs_new is None:
            self.fs_new = self.fs_old
        self.meta.insert("fs", self.fs_new)

        kwargs = dict(fs_old=self.fs_old, fs_new=self.fs_new)

        if self.fs_old != self.fs_new:
            # Only compute indexes and outputs if the sample rate has changed
            for key, node in parent.index.items():
                self.index.add_output(
                    key=key,
                    func=PartitionByTrial(resample_metadata),
                    kwargs=dict(index=parent.index["index"], data=node, **kwargs,),
                )

            for key, node in parent.outputs.items():
                self.outputs.add_output(
                    key=key,
                    func=PartitionByTrial(resample_data),
                    kwargs=dict(index=parent.index["index"], data=node, **kwargs,),
                )

    @property
    def identifier(self):
        return self.parent.identifier / f"{self.fs_new}Hz"
