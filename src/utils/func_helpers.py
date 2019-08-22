import pandas as pd
from tqdm import tqdm
from ..utils import sliding_window_rect

__all__ = [
    'Partition', 'PartitionAndWindow'
]


class Partition(object):
    def __init__(self, func):
        self.func = func
        setattr(self, '__name__', func.__name__)
    
    def __call__(self, key, index, data, *args, **kwargs):
        output = []
        trials = index.trial.unique()
        df_output = []
        for trial in tqdm(trials):
            inds = index.trial == trial
            vals = self.func(
                key=key,
                index=index.loc[inds],
                data=data.loc[inds],
                *args,
                **kwargs
            )
            if isinstance(vals, pd.DataFrame):
                output.append(vals)
                df_output.append(True)
            else:
                output.extend(vals)
                df_output.append(False)
        assert len(set(df_output)) == 1
        if all(df_output):
            df = pd.concat(output, axis=0)
        else:
            df = pd.DataFrame(output)
        return df


class PartitionAndWindow(Partition):
    def __init__(self, func, win_len, win_inc):
        def partitioned_func(key, index, data, *args, **kwargs):
            index_ = sliding_window_rect(index.values, win_len, win_inc)
            data_ = sliding_window_rect(data.values, win_len, win_inc)
            return func(
                key=key,
                index=index_,
                data=data_,
                *args,
                **kwargs
            )
        
        super(PartitionAndWindow, self).__init__(
            partitioned_func
        )

# class PartitionWindowExtract(object):
#     def __init__(self, win_len, win_inc, feat_func):
#         self.win_len = win_len
#         self.win_inc = win_inc
#
#         self.feat_func = feat_func
#
#         setattr(self, '__name__', feat_func.__name__)
#
#     def __call__(self, key, index, data, *args, **kwargs):
#         assert index.shape[0] == data.shape[0]
#
#         t0, t1 = index.time.values[[0, 1]]
#         print(t0, t1)
#         fs = 1.0 / (t1 - t0)
#         print(fs)
#
#         win_len = int(self.win_len * fs)
#         win_inc = int(self.win_inc * fs)
#
#         output = []
#
#         data = data.copy()
#         if key == 'label':
#             lookup, lookdown = get_transformer(data)
#             data = transform_df(data, lookup)
#
#         trials = index.trial.unique()
#         for trial in tqdm(trials, 'Trial'):
#             inds = index.trial == trial
#             w_index = sliding_window_rect(index.loc[inds].values, win_len, win_inc)
#             w_data = sliding_window_rect(data.loc[inds].values, win_len, win_inc)
#             if data.shape[1] == 1:
#                 w_data = w_data[..., None]
#             feat = self.feat_func(key=key, index=w_index, data=w_data, *args, **kwargs)
#             output.extend(feat)
#
#         df = pd.DataFrame(output)
#         if key == 'label':
#             df = transform_df(df, lookdown)
#         return df
#
# class PartitionWindowExtractMetaData(PartitionWindowExtract):
#     def __init__(self, win_len, win_inc):
#         def medianifier(key, index, data, *args, **kwargs):
#             assert key in {'label', 'index', 'fold'}
#             assert data.ndim == 3
#             return np.median(data, axis=1)
#
#         super(PartitionWindowExtractMetaData, self).__init__(
#             win_len=win_len,
#             win_inc=win_inc,
#             feat_func=medianifier
#         )
