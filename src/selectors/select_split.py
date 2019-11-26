import pandas as pd

from src.selectors.base import SelectorBase

__all__ = [
    'select_split',
]


def as_tr_vl_te(df):
    for col in df.columns:
        if not isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].apply(lambda v: ['train', 'test'][v])
    return df.astype('category')


def predefined_split(key, split):
    return as_tr_vl_te(split)


def loso_split(key, split):
    trials = sorted(split.subject.unique().astype(int))
    return as_tr_vl_te(pd.DataFrame(
        {kk: split.trial == kk for kk in trials}
    ).astype(int))


class select_split(SelectorBase):
    def __init__(self, parent, split_type):
        super(select_split, self).__init__(
            name=split_type, parent=parent, meta='split.yaml'
        )
        
        assert split_type in self.meta['supported']
        
        self.index.add_output(
            key='split',
            func=dict(
                predefined=predefined_split,
                loso=loso_split,
            )[split_type],
            split=dict(
                predefined=parent.index.fold,
                loso=parent.index.index,
            )[split_type],
            backend='pandas',
        )
