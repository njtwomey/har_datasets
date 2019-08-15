from os.path import join

from numpy import ceil, arange
from scipy.signal import resample

from src import BaseProcessor


def resampler(df, fs, fs_new):
    n_samples = ceil(df.shape[0] * (fs_new / fs))
    t = arange(df.shape[0]) / fs
    
    x_new, t_new = resample(
        x=df.values,
        num=n_samples,
        t=t,
        axis=0
    )
    
    return t_new, x_new


class Resampler(BaseProcessor):
    def __init__(self, name, parent, fs_new):
        super(Resampler, self).__init__(name)
        self.parent = parent,
        self.fs_new = fs_new
    
    def compose(self):
        for k1, v1 in self.parent.outputs.items():
            for k2, v2 in v1.items():
                self.outputs[k1][k2] = self.node(
                    node_name=join(self.dataset.processed_path, str(self.new_fs), f'{k1}_{k2}'),
                    sources=dict(df=self.parent.outputs[k1][k2]),
                    kwargs=dict(fs=self.dataset.meta['fs'], fs_new=self.fs_new)
                )
