from os.path import join

import pandas as pd
import numpy as np

from scipy.io import loadmat

from tqdm import trange

from src.utils.decorators import index_decorator, label_decorator, fold_decorator
from src.datasets.base import Dataset

__all__ = ["uschad"]


class uschad(Dataset):
    def __init__(self):
        super(uschad, self).__init__(name=self.__class__.__name__,)

    @label_decorator
    def build_label(self, task, *args, **kwargs):
        def callback(ii, sub_id, act_id, trial_id, data):
            return np.zeros((data.shape[0], 1)) + act_id

        return (
            self.meta.inv_lookup[task],
            uschad_iterator(self.unzip_path, callback=callback, desc=f"{self.identifier} Labels"),
        )

    @fold_decorator
    def build_fold(self, *args, **kwargs):
        def callback(ii, sub_id, act_id, trial_id, data):
            return np.tile(["train", "test"][sub_id > 10], (data.shape[0], 1))

        return uschad_iterator(self.unzip_path, callback=callback, desc=f"{self.identifier} Folds")

    @index_decorator
    def build_index(self, *args, **kwargs):
        def callback(ii, sub_id, act_id, trial_id, data):
            return np.c_[
                np.zeros((data.shape[0], 1)) + sub_id,
                np.zeros((data.shape[0], 1)) + ii,
                np.arange(data.shape[0]) / self.meta["fs"],
            ]

        return uschad_iterator(
            self.unzip_path,
            callback=callback,
            columns=["subject", "trial", "time"],
            desc=f"{self.identifier} Index",
        )

    def build_data(self, key, *args, **kwargs):
        modality, placement = key
        cols = dict(accel=[0, 1, 2], gyro=[3, 4, 5])[modality]

        def callback(ii, sub_id, act_id, trial_id, data):
            return data[:, cols]

        scale = dict(accel=1.0, gyro=2 * np.pi / 360)

        data = uschad_iterator(
            self.unzip_path, callback=callback, desc=f"Data ({modality}-{placement})"
        )

        return data.values * scale[modality]


def uschad_iterator(path, columns=None, cols=None, callback=None, desc=None):
    data_list = []

    ii = 0

    for sub_id in trange(1, 14 + 1, desc=desc):
        for act_id in range(1, 12 + 1):
            for trail_id in range(1, 5 + 1):
                fname = join(path, f"Subject{sub_id}", f"a{act_id}t{trail_id}.mat")

                data = loadmat(fname)["sensor_readings"]

                if callback:
                    data = callback(ii, sub_id, act_id, trail_id, data)
                elif cols:
                    data = data[cols]
                else:
                    raise ValueError

                data_list.extend(data)
                ii += 1

    df = pd.DataFrame(data_list)
    if columns:
        df.columns = columns
    return df
