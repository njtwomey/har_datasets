from os.path import join

import numpy as np
import pandas as pd

from src.datasets.base import Dataset
from src.utils.decorators import fold_decorator
from src.utils.decorators import index_decorator
from src.utils.decorators import label_decorator
from src.utils.loaders import load_csv_data

__all__ = ["anguita2013"]

WIN_LEN = 128


class anguita2013(Dataset):
    def __init__(self):
        super(anguita2013, self).__init__(name=self.__class__.__name__, unzip_path=lambda s: s.replace("%20", " "))

    @label_decorator
    def build_label(self, task, *args, **kwargs):
        labels = []
        for fold in ("train", "test"):
            fold_labels = load_csv_data(join(self.unzip_path, fold, f"y_{fold}.txt"))
            labels.extend([l for l in fold_labels for _ in range(WIN_LEN)])
        return self.meta.inv_lookup[task], pd.DataFrame(dict(labels=labels))

    @fold_decorator
    def build_predefined(self, *args, **kwargs):
        fold = []
        fold.extend(
            ["train" for _ in load_csv_data(join(self.unzip_path, "train", "y_train.txt")) for _ in range(WIN_LEN)]
        )
        fold.extend(
            ["test" for _ in load_csv_data(join(self.unzip_path, "test", "y_test.txt")) for _ in range(WIN_LEN)]
        )
        return pd.DataFrame(fold)

    @index_decorator
    def build_index(self, *args, **kwargs):
        sub = []
        for fold in ("train", "test"):
            sub.extend(load_csv_data(join(self.unzip_path, fold, f"subject_{fold}.txt")))
        index = pd.DataFrame(
            dict(
                subject=[si for si in sub for _ in range(WIN_LEN)],
                trial=build_seq_list(subs=sub, win_len=WIN_LEN),
                time=build_time(subs=sub, win_len=WIN_LEN, fs=float(self.meta.meta["fs"])),
            )
        )
        return index

    def build_data(self, loc, mod, *args, **kwargs):
        x_data = []
        y_data = []
        z_data = []
        modality = dict(accel="acc", gyro="gyro")[mod]
        for fold in ("train", "test"):
            for l, d in zip((x_data, y_data, z_data), ("x", "y", "z")):
                ty = ["body", "total"][modality in {"accel", "acc"}]
                acc = load_csv_data(
                    join(self.unzip_path, fold, "Inertial Signals", f"{ty}_{modality}_{d}_{fold}.txt"), astype="np",
                )
                l.extend(acc.ravel().tolist())
        return np.c_[x_data, y_data, z_data]


def build_time(subs, win_len, fs):
    win = np.arange(win_len, dtype=float) / fs
    inc = win_len / fs
    t = []
    prev_sub = subs[0]
    for curr_sub in subs:
        if curr_sub != prev_sub:
            win = np.arange(win_len, dtype=float) / fs
        t.extend(win)
        win += inc
        prev_sub = curr_sub
    return t


def build_seq_list(subs, win_len):
    seq = []
    si = 0
    last_sub = subs[0]
    for prev_sub in subs:
        if prev_sub != last_sub:
            si += 1
        seq.extend([si] * win_len)
        last_sub = prev_sub
    return seq
