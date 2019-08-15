import pandas as pd
from os.path import join

from src.utils import load_data, build_time, build_seq_list

from ..dataset_processor import DatasetProcessor


class anguita2013(DatasetProcessor):
    def __init__(self):
        super(anguita2013, self).__init__(
            name='anguita2013',
            unzip_path=lambda s: s.replace('%20', ' ')
        )
        
        self.win_len = 128
    
    def build_labels(self, path, *args, **kwargs):
        labels = []
        for fold in ('train', 'test'):
            fold_labels = load_data(join(path, fold, f'y_{fold}.txt'))
            labels.extend([self.dataset.inv_lookup[l] for l in fold_labels for _ in range(self.win_len)])
        labels = pd.DataFrame(dict(labels=labels)).astype('category')
        return labels
    
    def build_folds(self, path, *args, **kwargs):
        fold = []
        fold.extend([-1 for _ in load_data(join(path, 'train', 'y_train.txt')) for _ in range(self.win_len)])
        fold.extend([+1 for _ in load_data(join(path, 'test', 'y_test.txt')) for _ in range(self.win_len)])
        fold = pd.DataFrame(dict(fold=fold)).astype(int)
        return fold
    
    def build_index(self, path, *args, **kwargs):
        labels = []
        sub = []
        for fold in ('train', 'test'):
            sub.extend(load_data(join(path, fold, f'subject_{fold}.txt')))
            labels.extend(load_data(join(path, fold, f'y_{fold}.txt')))
        index = pd.DataFrame(dict(
            sub=[si for si in sub for _ in range(self.win_len)],
            sub_seq=build_seq_list(subs=sub, win_len=self.win_len),
            time=build_time(subs=sub, win_len=self.win_len, fs=float(self.dataset.meta['fs'])),
        )).astype(dict(
            sub=int,
            sub_seq=int,
            time=float
        ))
        return index
    
    def build_source(self, path, modality, location, *args, **kwargs):
        x_data = []
        y_data = []
        z_data = []
        modality = dict(
            accel='acc',
            gyro='gyro',
        )[modality]
        for fold in ('train', 'test'):
            for l, d in zip((x_data, y_data, z_data), ('x', 'y', 'z')):
                a = load_data(join(path, fold, 'Inertial Signals', f'body_{modality}_{d}_{fold}.txt'), astype='np')
                if modality in {'accel', 'acc'}:
                    b = load_data(join(path, fold, 'Inertial Signals', f'total_{modality}_{d}_{fold}.txt'), astype='np')
                    l.extend((a.ravel() + b.ravel()).tolist())
                else:
                    l.extend(a.ravel().tolist())
        data = pd.DataFrame(dict(x=x_data, y=y_data, z=z_data)).astype(float)
        return data
