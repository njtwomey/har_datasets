from collections import defaultdict
from os.path import join

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.datasets.base import Dataset
from src.utils.decorators import fold_decorator
from src.utils.decorators import index_decorator
from src.utils.decorators import label_decorator

__all__ = [
    "pamap2",
]


class pamap2(Dataset):
    def __init__(self):
        super(pamap2, self).__init__(name=self.__class__.__name__, unzip_path=lambda p: join(p, "Protocol"))

    @label_decorator
    def build_label(self, task, *args, **kwargs):
        df = pd.DataFrame(iter_pamap2_subs(path=self.unzip_path, cols=[1], desc=f"{self.identifier} Labels"))

        return self.meta.inv_lookup[task], df

    @fold_decorator
    def build_predefined(self, *args, **kwargs):
        def folder(sid, data):
            return np.zeros(data.shape[0]) + sid

        df = iter_pamap2_subs(
            path=self.unzip_path, cols=[1], desc=f"{self.identifier} Folds", callback=folder, columns=["fold"],
        ).astype(int)

        lookup = {
            1: "train",
            2: "train",
            3: "test",
            4: "train",
            5: "train",
            6: "test",
            7: "train",
            8: "train",
            9: "test",
        }

        return df.assign(fold_0=df["fold"].apply(lookup.__getitem__))[["fold_0"]].astype("category")

    @index_decorator
    def build_index(self, *args, **kwargs):
        def indexer(sid, data):
            subject = np.zeros(data.shape[0])[:, None] + sid
            trial = np.zeros(data.shape[0])[:, None] + sid
            return np.concatenate((subject, trial, data), axis=1)

        df = iter_pamap2_subs(
            path=self.unzip_path,
            cols=[0],
            desc=f"{self.identifier} Index",
            callback=indexer,
            columns=["subject", "trial", "time"],
        ).astype(dict(subject=int, trial=int, time=float))

        return df

    def build_data(self, loc, mod, *args, **kwargs):
        offset = dict(wrist=3, chest=20, ankle=37)[loc] + dict(accel=1, gyro=7, mag=10)[mod]

        df = iter_pamap2_subs(
            path=self.unzip_path,
            cols=list(range(offset, offset + 3)),
            desc=f"Parsing {mod} at {loc}",
            columns=["x", "y", "z"],
        ).astype(float)

        scale = dict(accel=9.80665, gyro=np.pi * 2.0, mag=1.0)[mod]

        return df.values / scale


def iter_pamap2_subs(path, cols, desc, columns=None, callback=None, n_subjects=9):
    data = []

    for sid in tqdm(range(1, n_subjects + 1), desc=desc):
        datum = pd.read_csv(join(path, f"subject10{sid}.dat"), delim_whitespace=True, header=None, usecols=cols).fillna(
            method="ffill"
        )
        assert np.isfinite(datum.values).all()
        if callback:
            data.extend(callback(sid, datum.values))
        else:
            data.extend(datum.values)
    df = pd.DataFrame(data)
    if columns:
        df.columns = columns
    return df
