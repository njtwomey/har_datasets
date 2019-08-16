from os.path import join, splitext, basename

from os import environ

from collections import defaultdict

import yaml

from mldb import ComputationGraph, PickleBackend, VolatileBackend

from .utils import check_activities, check_modalities, check_locations, unzip_data
from .utils import index_decorator, label_decorator, fold_decorator, data_decorator

__all__ = [
    'DatasetMeta', 'BaseDataset', 'Dataset'
]


class DatasetMeta(object):
    def __init__(self, dataset):
        if not isinstance(dataset, dict):
            datasets = yaml.load(open(join(
                environ['PROJECT_ROOT'], 'datasets.yaml'
            ), 'r'))
            assert dataset in datasets
            dataset = datasets[dataset]
        
        self.name = dataset['name']
        self.meta = dataset
        
        self.root = join(
            environ['PROJECT_ROOT'],
            'data'
        )
        
        self.inv_lookup = check_activities(self.meta['activities'])
        
        self.labels = self.meta['activities']
        self.locations = self.meta['locations'].keys()
        self.modalities = self.meta['modalities']
        
        check_activities(self.labels)
        check_locations(self.locations)
        check_modalities(self.modalities)
    
    @property
    def data_path(self):
        return join(self.root, 'data', self.name)
    
    @property
    def zip_path(self):
        return join(self.root, 'zip', self.name)
    
    @property
    def raw_path(self):
        return join(self.root, 'raw', self.name)
    
    @property
    def processed_path(self):
        return join(self.root, 'processed', self.name)
    
    @property
    def features_path(self):
        return join(self.root, 'features', self.name)
    
    @property
    def model_path(self):
        return join(self.root, 'models', self.name)
    
    def make_row(self):
        def make_links(links, desc='Link'):
            return ', '.join('[{} {}]({})'.format(desc, ii, url) for ii, url in enumerate(links, start=1))
        
        # modalities = sorted(set([mn for ln, lm in self.meta['locations'].items() for mn, mv in lm.items() if mv]))
        
        data = [
            self.meta['author'],
            self.meta['paper_name'],
            self.name,
            make_links(links=self.meta['description_urls'], desc='Link'),
            make_links(links=self.meta['paper_urls'], desc='Link'),
            self.meta['year'],
            self.meta['fs'],
            ', '.join(self.meta['locations'].keys()),
            ', '.join(self.meta['modalities']),
            self.meta['num_subjects'],
            self.meta['num_activities'],
            ', '.join(self.meta['activities'].keys()),
        ]
        
        return (
            (
                f'| First Author | Paper Name | Dataset Name | Description '
                f'| Download Links | Year | Sampling Rate | Device Locations | Device Modalities '
                f'| Num Subjects | Num Activities | Activities | '
            ),
            '| {} |'.format(' | '.join(['-----'] * len(data))),
            '| {} |'.format(' | '.join(map(str, data)))
        )


class BaseDataset(ComputationGraph):
    """
    A simple computational graph that is meant only to define backends and load metadata
    """
    
    def __init__(self, name):
        super(BaseDataset, self).__init__(
            name=name,
            default_backend='fs',
        )
        
        self.dataset = DatasetMeta(self.name)
        self.fs_root = join(self.dataset.root)
        
        self.add_backend('fs', PickleBackend(self.fs_root))
        self.add_backend('none', VolatileBackend())


class Dataset(BaseDataset):
    """
    A dataset has labels, index, fold definitions, and outputs defined
    """
    
    def __init__(self, name, *args, **kwargs):
        super(Dataset, self).__init__(name=name)
        
        self.labels = None
        self.fold = None
        self.index = None
        self.outputs = None
        
        zip_name = basename(self.dataset.meta['download_urls'][0])
        unzip_path = join(self.dataset.zip_path, splitext(zip_name)[0])
        
        self.unzip_path = kwargs.get('unzip_path', lambda x: x)(unzip_path)
        
        self.compose()
        
        self.node(
            node_name='unzipped',
            func=unzip_data,
            kwargs=dict(
                zip_path=self.dataset.zip_path,
                in_name=zip_name,
                out_name=self.unzip_path
            ),
            backend='none'
        )
    
    def compose(self):
        self.labels = self.node(
            node_name=join(self.dataset.raw_path, 'labels'),
            func=self.build_labels,
            kwargs=dict(
                path=self.unzip_path,
                inv_lookup=self.dataset.inv_lookup
            )
        )
        
        self.fold = self.node(
            node_name=join(self.dataset.raw_path, 'fold'),
            func=self.build_folds,
            kwargs=dict(
                path=self.unzip_path,
            )
        )
        
        self.index = self.node(
            node_name=join(self.dataset.raw_path, 'index'),
            func=self.build_index,
            kwargs=dict(
                path=self.unzip_path
            )
        )
        
        self.outputs = {kk: dict() for kk in self.dataset.modalities}
        
        for location, amga in self.dataset.meta['locations'].items():
            for acc_mag_gyr, active in amga.items():
                if not active:
                    continue
                self.outputs[acc_mag_gyr][location] = self.node(
                    node_name=join(self.dataset.raw_path, f'{acc_mag_gyr}-{location}'),
                    func=self.build_data,
                    sources=None,
                    kwargs=dict(
                        path=self.unzip_path,
                        modality=acc_mag_gyr,
                        location=location,
                    )
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
