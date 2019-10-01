# For numpy serialisation
from numpy import load as np_load
from numpy import save as np_save
from numpy import ndarray

# For pandas serialisation
from pandas import DataFrame
from pandas import read_pickle as pd_load

# For sklearn serialisation
import joblib

from mldb import FileSystemBase

__all__ = [
    'PNGBackend', 'ScikitLearnBackend', 'NumpyBackend', 'PandasBackend',
]


class PNGBackend(FileSystemBase):
    def __init__(self, path):
        super(PNGBackend, self).__init__(path, 'png')
    
    def load_data(self, name):
        return True
    
    def save_data(self, name, data):
        fig = data
        fig.savefig(self.node_path(name))
        fig.clf()


class PandasBackend(FileSystemBase):
    def __init__(self, path, compression='gzip'):
        super(PandasBackend, self).__init__(path, 'pd')
        self.compression = compression
    
    def load_data(self, name):
        data = pd_load(
            path=self.node_path(name),
            compression=self.compression,
        )
        assert isinstance(data, DataFrame)
        return data
    
    def save_data(self, name, data):
        assert isinstance(data, DataFrame)
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
        assert isinstance(data, ndarray)
        return data
    
    def save_data(self, name, data):
        assert isinstance(data, ndarray)
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
        return model
    
    def save_data(self, name, data):
        joblib.dump(data, self.node_path(name))
