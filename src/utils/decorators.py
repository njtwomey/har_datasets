import pandas as pd
from numpy import isfinite

from functools import update_wrapper, partial

__all__ = [
    'index_decorator', 'fold_decorator', 'label_decorator', 'data_decorator',
    'transformer_decorator', 'feature_decorator', 'model_decorator',
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
        return pd.DataFrame(df)


class LabelDecorator(DecoratorBase):
    def __init__(self, func):
        super(LabelDecorator, self).__init__(func)
    
    def __call__(self, *args, **kwargs):
        df = super(LabelDecorator, self).__call__(*args, **kwargs)
        if isinstance(df, tuple):
            inv_lookup, df = df
            df = pd.DataFrame(df)
            for ci in df.columns:
                df[ci] = df[ci].apply(
                    lambda ll: inv_lookup[ll]
                )
        df = pd.DataFrame(df).astype('category')
        df.columns = [f'track_{fi}' for fi in range(len(df.columns))]
        return df


class FoldDecorator(DecoratorBase):
    def __init__(self, func):
        super(FoldDecorator, self).__init__(func)
    
    def __call__(self, *args, **kwargs):
        df = super(FoldDecorator, self).__call__(*args, **kwargs).astype(int)
        df.columns = [f'fold_{fi}' for fi in range(len(df.columns))]
        return df


class IndexDecorator(DecoratorBase):
    def __init__(self, func):
        super(IndexDecorator, self).__init__(func)
    
    def __call__(self, *args, **kwargs):
        df = super(IndexDecorator, self).__call__(*args, **kwargs)
        df.columns = ['subject', 'trial', 'time']
        return df.astype(dict(
            subject=int,
            trial=int,
            time=float
        ))


class DataDecorator(DecoratorBase):
    def __init__(self, func):
        super(DataDecorator, self).__init__(func)
    
    def __call__(self, *args, **kwargs):
        df = super(DataDecorator, self).__call__(*args, **kwargs)
        assert isfinite(df.values).all(), f'Error evaluating {self.func}, {self.func.__name__}'
        return df


class TransformerDecorator(DataDecorator):
    def __init__(self, func):
        super(TransformerDecorator, self).__init__(func)


class FeatureDecorator(DataDecorator):
    def __init__(self, func):
        super(FeatureDecorator, self).__init__(func)


class ModelDecorator(DataDecorator):
    def __init__(self, func):
        super(ModelDecorator, self).__init__(func)


label_decorator = LabelDecorator
index_decorator = IndexDecorator
fold_decorator = FoldDecorator
data_decorator = DataDecorator
transformer_decorator = TransformerDecorator
feature_decorator = FeatureDecorator
model_decorator = ModelDecorator
