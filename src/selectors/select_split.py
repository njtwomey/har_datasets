import pandas as pd
from pathlib import Path
from src.selectors.base import SelectorBase

__all__ = [
    'select_split',
]


def predefined_split(key, split):
    return split.astype('category')


def deployable_split(key, split):
    return pd.DataFrame(
        {'fold_0': ['train'] * len(split)}, dtype='category'
    )


def loso_split(key, split):
    return pd.DataFrame({
        f'fold_{kk}': split.trial.apply(
            lambda tt: ['train', 'test'][tt == kk]
        ) for kk in split.trial.unique()
    })


class select_split(SelectorBase):
    def __init__(self, parent, split_type):
        super(select_split, self).__init__(
            name=split_type, parent=parent, meta=Path('metadata', 'split.yaml')
        )

        assert split_type in self.meta['supported']

        func_dict = dict(
            predefined=predefined_split,
            deployable=deployable_split,
            loso=loso_split,
        )

        split = parent.index['index']
        if split_type == 'predefined':
            split = parent.index['fold']

        self.index.add_output(
            key='split',
            func=func_dict[split_type],
            backend='pandas',
            kwargs=dict(
                split=split,
            )
        )
