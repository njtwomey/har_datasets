from ..base import Dataset
from ..utils import index_decorator, label_decorator, fold_decorator, data_decorator


class dataset_name_goes_here(Dataset):
    def __init__(self):
        super(dataset_name_goes_here, self).__init__(
            name='dataset_name_goes_here',
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
