import pandas as pd
import numpy as np

from tqdm import tqdm

from functools import update_wrapper, partial

from src.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    'index_decorator', 'fold_decorator', 'label_decorator',
    'Partition', 'partition',
]


class DecoratorBase(object):
    def __init__(self, func):
        update_wrapper(self, func)
        self.func = func
    
    def __get__(self, obj, objtype):
        return partial(self.__call__, obj)
    
    def __call__(self, *args, **kwargs):
        df = self.func(*args, **kwargs)
        if isinstance(df, tuple):
            assert len(df) == 2
            return df
        return df


class LabelDecorator(DecoratorBase):
    def __init__(self, func):
        super(LabelDecorator, self).__init__(func)
    
    def __call__(self, *args, **kwargs):
        df = super(LabelDecorator, self).__call__(*args, **kwargs)
        
        # TODO/FIXME: remove this strange pattern
        if isinstance(df, tuple):
            inv_lookup, df = df
            df = pd.DataFrame(df)
            for ci in df.columns:
                df[ci] = df[ci].apply(
                    lambda ll: inv_lookup[ll]
                )
        
        df = pd.DataFrame(df)
        df.columns = [f'track_{fi}' for fi in range(len(df.columns))]
        df = df.astype({col: 'category' for col in df.columns})
        return df


class FoldDecorator(DecoratorBase):
    def __init__(self, func):
        super(FoldDecorator, self).__init__(func)
    
    def __call__(self, *args, **kwargs):
        df = super(FoldDecorator, self).__call__(*args, **kwargs)
        if isinstance(df.columns, pd.RangeIndex):
            df.columns = [f'fold_{fi}' for fi in range(len(df.columns))]
        df = df.astype({col: 'category' for col in df.columns})
        return df


class IndexDecorator(DecoratorBase):
    def __init__(self, func):
        super(IndexDecorator, self).__init__(func)
    
    def __call__(self, *args, **kwargs):
        df = super(IndexDecorator, self).__call__(*args, **kwargs)
        df.columns = ['subject', 'trial', 'time']
        return df.astype(dict(
            subject='category',
            trial='category',
            time=float
        ))


def infer_data_type(data):
    """

    Args:
        data:

    Returns:

    """
    if isinstance(data, np.ndarray):
        return 'numpy'
    elif isinstance(data, pd.DataFrame):
        return 'pandas'
    logger.exception(f"Unsupported data type ({type(data)}), currently only {{numpy, pandas}}")
    raise TypeError


class Partition(object):
    """

    """
    
    def __init__(self, func):
        """

        Args:
            func:
        """
        self.func = func
        update_wrapper(self, func)
        setattr(self, '__name__', func.__name__)
    
    def __get__(self, obj, objtype):
        return partial(self.__call__, obj)
    
    def __call__(self, key, index, data, *args, **kwargs):
        """

        Args:
            key:
            index:
            data:
            *args:
            **kwargs:

        Returns:

        """
        assert index.shape[0] == data.shape[0]
        output = []
        trials = index.trial.unique()
        data_type = infer_data_type(data)
        for trial in tqdm(trials):
            inds = index.trial == trial
            index_ = index.loc[inds]
            if data_type == 'numpy':
                data_ = data[inds]
            elif data_type == 'pandas':
                data_ = data.loc[inds]
            assert index_.shape[0] == data_.shape[0]
            vals = self.func(
                key=key,
                index=index_,
                data=data_,
                *args,
                **kwargs
            )
            assert infer_data_type(vals) == data_type
            output.append(vals)
        if data_type == 'numpy':
            df = np.concatenate(output, axis=0)
        elif data_type == 'pandas':
            df = pd.concat(output, axis=0)
            df = df.reset_index(drop=True)
        return df


label_decorator = LabelDecorator
index_decorator = IndexDecorator
fold_decorator = FoldDecorator
partition = Partition
