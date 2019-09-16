import pandas as pd
import numpy as np

from tqdm import tqdm

__all__ = [
    'Partition',
]


class Partition(object):
    def __init__(self, func):
        self.func = func
        setattr(self, '__name__', func.__name__)
    
    def __call__(self, key, index, data, *args, **kwargs):
        assert index.shape[0] == data.shape[0]
        if not all(data.index == index.index):
            index = index.reset_index(drop=True)
            data = data.reset_index(drop=True)
        output = []
        trials = index.trial.unique()
        df_output = []
        assert data.notna().all().all()
        for trial in tqdm(trials):
            inds = index.trial == trial
            index_ = index.loc[inds]
            data_ = data.loc[inds]
            assert index_.shape[0] == data_.shape[0]
            vals = self.func(
                key=key,
                index=index_,
                data=data_,
                *args,
                **kwargs
            )
            output.append(vals)
            if isinstance(vals, pd.DataFrame):
                df_output.append(True)
            else:
                df_output.append(False)
        assert len(set(df_output)) == 1
        if all(df_output):
            df = pd.concat(output, axis=0)
        else:
            try:
                df = pd.DataFrame(
                    np.concatenate(output, axis=0)
                )
            except ValueError:
                return np.concatenate(output, axis=0)
        return df.reset_index(drop=True)

# from . import sliding_window_rect
#
# class PartitionAndWindow(Partition):
#     def __init__(self, func, win_len, win_inc):
#         def windowed_func(key, index, data, *args, **kwargs):
#             win_len_ = int(win_len * kwargs['fs'])
#             win_inc_ = int(win_inc * kwargs['fs'])
#
#             index_ = sliding_window_rect(
#                 index.values, win_len_, win_inc_
#             )
#
#             data_ = sliding_window_rect(
#                 data.values, win_len_, win_inc_
#             )
#
#             assert index_.shape[0] == data_.shape[0]
#             assert index_.shape[1] == data_.shape[1]
#
#             ret = func(
#                 key=key,
#                 index=index_,
#                 data=data_,
#                 *args,
#                 **kwargs
#             )
#
#             return ret
#
#         super(PartitionAndWindow, self).__init__(
#             windowed_func
#         )
#
#
# class PartitionAndWindowMetadata(PartitionAndWindow):
#     def __init__(self, win_len, win_inc):
#         def median_filter(key, index, data, *args, **kwargs):
#             assert key in {'label', 'index', 'fold'}
#             if key == 'label':
#                 vals, transformed = np.unique(data, return_inverse=True)
#                 transformed = transformed.reshape(data.shape)
#                 med = np.median(transformed, axis=1).astype(int)
#                 return vals[med]
#             return np.median(data, axis=1)
#
#         super(PartitionAndWindowMetadata, self).__init__(
#             func=median_filter,
#             win_len=win_len,
#             win_inc=win_inc,
#         )
