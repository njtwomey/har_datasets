from mldb import FileSystemBase

from sklearn.externals import joblib

__all__ = [
    'PNGBackend', 'ScikitLearnBackend'
]


class ScikitLearnBackend(FileSystemBase):
    def __init__(self, path):
        super(ScikitLearnBackend, self).__init__(path, 'sklearn')
    
    def load_data(self, node_name):
        model = joblib.load(node_name)
        return model
    
    def save_data(self, node_name, data):
        joblib.dump(data, node_name)


class PNGBackend(FileSystemBase):
    def __init__(self, path):
        super(PNGBackend, self).__init__(path, 'png')
    
    def load_data(self, node_name):
        return None
    
    def save_data(self, node_name, data):
        data.savefig(node_name)
        data.clf()
