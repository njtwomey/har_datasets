from os.path import join
from os import environ

from collections import defaultdict

import yaml

from mldb import ComputationGraph, PickleBackend, VolatileBackend

from .utils import check_activities


class Dataset(object):
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


class _BaseProcessor(ComputationGraph):
    def __init__(self, name):
        super(_BaseProcessor, self).__init__(
            name=name,
            default_backend='fs',
        )
        
        self.dataset = Dataset(self.name)
        self.fs_root = join(self.dataset.root)
        
        self.add_backend('fs', PickleBackend(self.fs_root))
        self.add_backend('none', VolatileBackend())


class BaseProcessor(_BaseProcessor):
    def __init__(self, name):
        super(BaseProcessor, self).__init__(name=name)
        
        self.dataset_labels = self.dataset.meta['activities']
        check_activities(self.dataset_labels)
        
        self.index = None
        self.folds = None
        self.labels = None
        
        self.outputs = defaultdict(dict)
        
    def compose(self):
        raise NotImplementedError


class CompositeProcessor(_BaseProcessor):
    def __init__(self, name, parent):
        super(CompositeProcessor, self).__init__(name=name)
        self.parent = parent
