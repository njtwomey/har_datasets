# For numpy serialisation
from numpy import load as np_load
from numpy import save as np_save
from numpy import ndarray

# For pandas serialisation
from pandas import DataFrame
from pandas import read_pickle as pd_load

# For sklearn serialisation
from sklearn.externals import joblib

from mldb import FileSystemBase

__all__ = [
    'PNGBackend', 'ScikitLearnBackend', 'NumpyBackend', 'PandasBackend',
]


class PNGBackend(FileSystemBase):
    def __init__(self, path):
        super(PNGBackend, self).__init__(path, 'png')
    
    def load_data(self, node_name):
        return True
    
    def save_data(self, node_name, data):
        fig = data
        fig.savefig(self.node_path(node_name))
        fig.clf()


class PandasBackend(FileSystemBase):
    def __init__(self, path):
        super(PandasBackend, self).__init__(path, 'pd')
        self.compression = 'gzip'
    
    def load_data(self, node_name):
        data = pd_load(
            path=self.node_path(node_name),
            compression=self.compression,
        )
        assert isinstance(data, DataFrame)
        return data
    
    def save_data(self, node_name, data):
        assert isinstance(data, DataFrame)
        data.to_pickle(
            path=self.node_path(node_name),
            compression=self.compression,
        )


class NumpyBackend(FileSystemBase):
    def __init__(self, path):
        super(NumpyBackend, self).__init__(path, 'npy')
        self.allow_pickle = True
    
    def load_data(self, node_name):
        data = np_load(
            file=self.node_path(node_name),
            allow_pickle=self.allow_pickle,
        )
        assert isinstance(data, ndarray)
        return data
    
    def save_data(self, node_name, data):
        assert isinstance(data, ndarray)
        np_save(
            file=self.node_path(node_name),
            arr=data,
            allow_pickle=self.allow_pickle,
        )


class ScikitLearnBackend(FileSystemBase):
    def __init__(self, path):
        super(ScikitLearnBackend, self).__init__(path, 'sklearn')
    
    def load_data(self, node_name):
        model = joblib.load(self.node_path(node_name))
        return model
    
    def save_data(self, node_name, data):
        joblib.dump(data, self.node_path(node_name))
