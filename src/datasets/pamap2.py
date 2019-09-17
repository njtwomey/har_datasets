import pandas as pd
import numpy as np
from collections import defaultdict
from os.path import join

from tqdm import tqdm

from ..utils import index_decorator, label_decorator, fold_decorator, data_decorator

from .base import Dataset

__all__ = [
    'pamap2',
]


def iter_pamap2_subs(path, cols, desc, columns=None, callback=None, n_subjects=9):
    data = []
    
    for sid in tqdm(range(1, n_subjects + 1), desc=desc):
        datum = pd.read_csv(
            join(path, f'subject10{sid}.dat'),
            delim_whitespace=True,
            header=None,
            usecols=cols
        ).fillna(method='ffill')
        assert np.isfinite(datum.values).all()
        if callback:
            data.extend(callback(sid, datum.values))
        else:
            data.extend(datum.values)
    df = pd.DataFrame(data)
    if columns:
        df.columns = columns
    return df


class pamap2(Dataset):
    def __init__(self):
        super(pamap2, self).__init__(
            name=self.__class__.__name__,
            unzip_path=lambda p: join(p, 'Protocol')
        )
    
    @label_decorator
    def build_label(self, *args, **kwargs):
        df = pd.DataFrame(iter_pamap2_subs(
            path=self.unzip_path,
            cols=[1],
            desc=f'{self.identifier} Labels'
        ))
        
        return self.meta.inv_act_lookup, df
    
    @fold_decorator
    def build_fold(self, *args, **kwargs):
        def folder(sid, data):
            return np.zeros(data.shape[0]) + sid
        
        df = iter_pamap2_subs(
            path=self.unzip_path,
            cols=[1],
            desc=f'{self.identifier} Folds',
            callback=folder,
            columns=['fold']
        ).astype(int)
        
        data = []
        cols = []
        for cat in df.fold.unique():
            lookup = defaultdict(lambda: 'train')
            lookup[cat] = 'test'
            data.append(df.fold.apply(lambda l: lookup[l]).values)
            cols.append(f'loo_{cat}')
        df = pd.DataFrame(
            np.asarray(data).T,
            columns=cols
        ).astype('category')
        return df
    
    @index_decorator
    def build_index(self, *args, **kwargs):
        def indexer(sid, data):
            subject = np.zeros(data.shape[0])[:, None] + sid
            trial = np.zeros(data.shape[0])[:, None] + sid
            return np.concatenate((
                subject, trial, data
            ), axis=1)
        
        df = iter_pamap2_subs(
            path=self.unzip_path,
            cols=[0],
            desc=f'{self.identifier} Index',
            callback=indexer,
            columns=['subject', 'trial', 'time']
        ).astype(dict(
            subject=int,
            trial=int,
            time=float
        ))
        
        return df
    
    @data_decorator
    def build_data(self, key, *args, **kwargs):
        modality, location = key
        offset = dict(
            wrist=3,
            chest=20,
            ankle=37
        )[location] + dict(
            accel=1,
            gyro=7,
            mag=10
        )[modality]
        
        df = iter_pamap2_subs(
            path=self.unzip_path,
            cols=list(range(offset, offset + 3)),
            desc=f'Parsing {modality} at {location}',
            columns=['x', 'y', 'z']
        ).astype(float)
        
        scale = 1
        if 'accel' in key:
            scale = 9.80665
        elif 'gyro' in key:
            scale = np.pi * 2
        elif 'mag' in key:
            scale = 1
        
        return df.values / scale
