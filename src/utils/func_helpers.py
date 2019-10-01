import pandas as pd
import numpy as np

from tqdm import tqdm

__all__ = [
    'Partition',
]


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
        setattr(self, '__name__', func.__name__)
    
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
            else:
                raise ValueError
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
        else:
            raise ValueError
        return df
