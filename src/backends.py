# For numpy serialisation
from numpy import load as np_load
from numpy import save as np_save
from numpy import ndarray

# For pandas serialisation
from pandas import DataFrame
from pandas import read_pickle as pd_load

# For matplotlib serialisation
from matplotlib.pyplot import Figure

# For sklearn serialisation
from sklearn.base import ClassifierMixin, TransformerMixin, BaseEstimator
import joblib

from mldb import FileSystemBase
from src.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    'PNGBackend', 'PDFBackend', 'ScikitLearnBackend', 'NumpyBackend', 'PandasBackend',
]


def validate_dtype(obj, dtypes):
    """
    
    Args:
        obj: input object
        dtypes: data type, or list of data types

    Returns: True if the type of obj is in the allowable set
    Raises: TypeError if obj is not
    """
    if isinstance(obj, dtypes):
        return
    logger.exception(f"The object {obj} is of the wrong type {type(object)} is not in {dtypes}")
    raise TypeError


class MatPlotLibBackend(FileSystemBase):
    def __init__(self, path, ext):
        super(MatPlotLibBackend, self).__init__(path, ext)
    
    def load_data(self, name):
        return True
    
    def save_data(self, name, data):
        fig = data
        validate_dtype(fig, Figure)
        fig.savefig(self.node_path(name))
        fig.clf()


class PNGBackend(MatPlotLibBackend):
    def __init__(self, path):
        super(PNGBackend, self).__init__(path, 'png')


class PDFBackend(MatPlotLibBackend):
    def __init__(self, path):
        super(PDFBackend, self).__init__(path, 'pdf')


class PandasBackend(FileSystemBase):
    def __init__(self, path, compression='gzip'):
        super(PandasBackend, self).__init__(path, 'pd')
        self.compression = compression
    
    def load_data(self, name):
        data = pd_load(
            path=self.node_path(name),
            compression=self.compression,
        )
        validate_dtype(data, DataFrame)
        return data
    
    def save_data(self, name, data):
        validate_dtype(data, DataFrame)
        data.to_pickle(
            path=self.node_path(name),
            compression=self.compression,
        )


class NumpyBackend(FileSystemBase):
    def __init__(self, path, allow_pickle=True):
        super(NumpyBackend, self).__init__(path, 'npy')
        self.allow_pickle = allow_pickle
    
    def load_data(self, name):
        data = np_load(
            file=self.node_path(name),
            allow_pickle=self.allow_pickle,
        )
        validate_dtype(data, ndarray)
        return data
    
    def save_data(self, name, data):
        validate_dtype(data, ndarray)
        np_save(
            file=self.node_path(name),
            arr=data,
            allow_pickle=self.allow_pickle,
        )


class ScikitLearnBackend(FileSystemBase):
    def __init__(self, path):
        super(ScikitLearnBackend, self).__init__(path, 'sklearn')
    
    def load_data(self, name):
        model = joblib.load(self.node_path(name))
        # validate_dtype(model, (ClassifierMixin, TransformerMixin, BaseEstimator))
        return model
    
    def save_data(self, name, data):
        # validate_dtype(data, (ClassifierMixin, TransformerMixin, BaseEstimator))
        joblib.dump(data, self.node_path(name))
