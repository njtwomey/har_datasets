import pandas as pd
import numpy as np

from os.path import join
from tqdm import tqdm

from ..base import Dataset
from ..utils import index_decorator, label_decorator, fold_decorator, data_decorator


def iter_pamap2_subs(path, cols, desc, columns=None, callback=None, n_subjects=9):
    data = []
    
    for sid in tqdm(range(1, n_subjects + 1), desc=desc):
        datum = pd.read_csv(
            join(path, f'subject10{sid}.dat'),
            delim_whitespace=True,
            header=None,
            usecols=cols
        )
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
            name='pamap2',
            unzip_path=lambda p: join(p, 'Protocol')
        )
    
    @label_decorator
    def build_labels(self, path, *args, **kwargs):
        df = pd.DataFrame(iter_pamap2_subs(
            path=path,
            cols=[1],
            desc='Labels'
        ))
        
        return self.dataset.inv_lookup, df
    
    @fold_decorator
    def build_folds(self, path, *args, **kwargs):
        df = iter_pamap2_subs(
            path=path,
            cols=[1],
            desc='Folds',
            callback=lambda sid, dd: np.zeros(dd.shape[0]) + sid,
            columns=['fold']
        ).astype(int)
        return df
    
    @index_decorator
    def build_index(self, path, *args, **kwargs):
        def indexer(sid, data):
            subject = np.zeros(data.shape[0])[:, None] + sid
            subject_seq = np.zeros(data.shape[0])[:, None] + sid
            return np.concatenate((
                subject, subject_seq, data
            ), axis=1)
        
        df = iter_pamap2_subs(
            path=path,
            cols=[0],
            desc='Index',
            callback=indexer,
            columns=['sub', 'sub_seq', 'time']
        ).astype(dict(
            sub=int,
            sub_seq=int,
            time=float
        ))
        return df
    
    @data_decorator
    def build_data(self, path, modality, location, *args, **kwargs):
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
            path=path,
            cols=list(range(offset, offset + 3)),
            desc=f'Parsing {modality} at {location}',
            columns=['x', 'y', 'z']
        ).astype(float) / 9.80665
        
        return df
