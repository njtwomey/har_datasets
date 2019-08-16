from ..base import Dataset
from ..utils import index_decorator, label_decorator, fold_decorator, data_decorator

__all__ = [
    'NameOfTheDataset'
]


class NameOfTheDataset(Dataset):
    def __init__(self):
        super(NameOfTheDataset, self).__init__(
            name=self.__class__.__name__,
        )
    
    @label_decorator
    def build_labels(self, path, *args, **kwargs):
        raise NotImplementedError
    
    @fold_decorator
    def build_folds(self, path, *args, **kwargs):
        raise NotImplementedError
    
    @index_decorator
    def build_index(self, path, *args, **kwargs):
        raise NotImplementedError
    
    @data_decorator
    def build_data(self, path, modality, location, *args, **kwargs):
        raise NotImplementedError
