from functools import partial
from functools import update_wrapper

import numpy as np
import pandas as pd
from loguru import logger
from pandas.api.types import is_categorical_dtype
from tqdm import tqdm

from src.utils.exceptions import ModalityNotPresentError
from src.utils.loaders import dataset_importer


__all__ = [
    "index_decorator",
    "fold_decorator",
    "label_decorator",
    "PartitionByTrial",
    "partitioning_decorator",
]


class DecoratorBase(object):
    def __init__(self, func):
        update_wrapper(self, func)
        self.func = func

    def __get__(self, obj, objtype):
        return partial(self.__call__, obj)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


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
                df[ci] = df[ci].apply(lambda ll: inv_lookup[ll])

        assert len(df.columns) == 1

        df = pd.DataFrame(df)
        df.columns = [f"target" for _ in range(len(df.columns))]
        if is_categorical_dtype(df["target"]):
            df = df.astype(dict(target="category"))

        return df


class FoldDecorator(DecoratorBase):
    def __init__(self, func):
        super(FoldDecorator, self).__init__(func)

    def __call__(self, *args, **kwargs):
        df = super(FoldDecorator, self).__call__(*args, **kwargs)
        if isinstance(df.columns, pd.RangeIndex):
            df.columns = [f"fold_{fi}" for fi in range(len(df.columns))]
        df = df.astype({col: "category" for col in df.columns})
        return df


class IndexDecorator(DecoratorBase):
    def __init__(self, func):
        super(IndexDecorator, self).__init__(func)

    def __call__(self, *args, **kwargs):
        df = super(IndexDecorator, self).__call__(*args, **kwargs)
        df.columns = ["subject", "trial", "time"]
        return df.astype(dict(subject="category", trial="category", time=float))


def infer_data_type(data):
    """

    Args:
        data:

    Returns:

    """
    if isinstance(data, np.ndarray):
        return "numpy"
    elif isinstance(data, pd.DataFrame):
        return "pandas"

    logger.exception(
        TypeError(
            f"Unsupported data type in infer_data_type ({type(data)}), currently only {{numpy, pandas}}"
        )
    )


def slice_data_type(data, inds, data_type_name):
    if data_type_name == "numpy":
        return data[inds]
    elif data_type_name == "pandas":
        return data.loc[inds]

    logger.exception(
        TypeError(
            f"Unsupported data type in slice_data_type ({type(data)}), currently only {{numpy, pandas}}"
        )
    )


def concat_data_type(datas, data_type_name):
    if data_type_name == "numpy":
        return np.concatenate(datas, axis=0)
    elif data_type_name == "pandas":
        df = pd.concat(datas, axis=0)
        return df.reset_index(drop=True)

    logger.exception(
        TypeError(
            f"Unsupported data type in concat_data_type ({type(datas)}), currently only {{numpy, pandas}}"
        )
    )


class PartitionByTrial(DecoratorBase):
    """

    """

    def __init__(self, func):
        super(PartitionByTrial, self).__init__(func=func)

    def __call__(self, index, data, *args, **kwargs):
        """

        Args:
            key:
            index:
            data:
            *args:
            **kwargs:

        Returns:

        """
        if index.shape[0] != data.shape[0]:
            logger.exception(
                ValueError(
                    f"The data and index  should have the same length "
                    "with index: {index.shape}; and data: {data.shape}"
                )
            )
        output = []
        trials = index.trial.unique()
        data_type = infer_data_type(data)
        for trial in tqdm(trials):
            inds = index.trial == trial
            index_ = index.loc[inds]
            data_ = slice_data_type(data, inds, data_type)
            vals = self.func(index=index_, data=data_, *args, **kwargs)
            opdt = infer_data_type(vals)
            if opdt != data_type:
                logger.exception(
                    ValueError(
                        f"The data type of {self.func} should be the same as the input {data_type} "
                        f"but instead got {opdt}"
                    )
                )
            output.append(vals)
        return concat_data_type(output, data_type)


class RequiredModalities(DecoratorBase):
    def __init__(self, func, *modalities):
        super(RequiredModalities, self).__init__(func=func)

        self.required_modalities = set(modalities)

    def __call__(self, dataset, *args, **kwargs):
        dataset = dataset_importer(dataset)
        dataset_modalities = dataset.meta.modalities
        for required_modality in self.required_modalities:
            if required_modality not in dataset_modalities:
                logger.exception(
                    ModalityNotPresentError(
                        f"The modality {required_modality} is required by the function {self.func}. "
                        f"However, the dataset {dataset} does not have {required_modality}. The "
                        f"available modalities are: {dataset_modalities})"
                    )
                )

        super(self, RequiredModalities).__call__(dataset, *args, **kwargs)


required_modalities = RequiredModalities


label_decorator = LabelDecorator
index_decorator = IndexDecorator
fold_decorator = FoldDecorator
partitioning_decorator = PartitionByTrial
