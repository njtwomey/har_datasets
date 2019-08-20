from os.path import join, splitext, basename
from os import environ

from collections import defaultdict

from mldb import ComputationGraph, PickleBackend, VolatileBackend

from .utils import check_activities, check_modalities, check_locations, unzip_data, load_yaml, build_path
from . import label_decorator, index_decorator, data_decorator, fold_decorator

__all__ = [
    'DatasetMeta', 'BaseGraph'
]


class DatasetMeta(object):
    def __init__(self, dataset):
        if not isinstance(dataset, dict):
            datasets = load_yaml('datasets.yaml')
            assert dataset in datasets
            dataset = datasets[dataset]
        
        self.name = dataset['name']
        self.meta = dataset
        
        self.fs = dataset['fs']
        
        self.root = build_path('data')
        
        self.inv_lookup = check_activities(self.meta['activities'])
        
        self.labels = self.meta['activities']
        self.locations = self.meta['locations'].keys()
        self.modalities = self.meta['modalities']
        
        check_activities(self.labels)
        check_locations(self.locations)
        check_modalities(self.modalities)
    
    @property
    def zip_path(self):
        return build_path('data', 'zip', self.name)
    
    @property
    def build_path(self):
        return build_path('data', 'build', self.name)
    
    def make_row(self):
        def make_links(links, desc='Link'):
            return ', '.join(
                '[{} {}]({})'.format(desc, ii, url) for ii, url in enumerate(links, start=1)
            )
        
        # modalities = sorted(set([mn for ln, lm in self.meta['locations'].items() for mn, mv in lm.items() if mv]))
        
        data = [
            self.meta['author'],
            self.meta['paper_name'],
            self.name,
            make_links(links=self.meta['description_urls'], desc='Link'),
            self.meta.get('missing', ''),
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
                f'| First Author | Paper Name | Dataset Name | Description | Missing data '
                f'| Download Links | Year | Sampling Rate | Device Locations | Device Modalities '
                f'| Num Subjects | Num Activities | Activities | '
            ),
            '| {} |'.format(' | '.join(['-----'] * len(data))),
            '| {} |'.format(' | '.join(map(str, data)))
        )


def walk_dict(dict_var, breadcrums=None):
    if breadcrums is None:
        breadcrums = tuple()
    for kk, vv in dict_var.items():
        if isinstance(vv, (dict, defaultdict)):
            yield from walk_dict(vv, breadcrums + (kk,))
        else:
            yield breadcrums + (kk,), vv


class BaseGraph(ComputationGraph):
    """
    A simple computational graph that is meant only to define backends and load metadata
    """
    
    def __init__(self, name):
        super(BaseGraph, self).__init__(
            name=name, default_backend='fs',
        )
        
        self.fs_root = build_path('data')
        
        self.add_backend('fs', PickleBackend(self.fs_root))
        self.add_backend('none', VolatileBackend())
        
        self.label = None
        self.fold = None
        self.index = None
        
        self.outputs = None
    
    def build_path(self, *args):
        return build_path('data', 'build', self.identifier, '-'.join(args))
    
    def iter_outputs(self):
        assert isinstance(self.outputs, (dict, defaultdict))
        for kk, vv in walk_dict(self.outputs):
            yield kk, vv
    
    def pre_compose(self):
        pass
    
    def compose(self):
        self.pre_compose()
        
        self.index = self.compose_index()
        assert self.index is not None
        
        self.fold = self.compose_fold()
        assert self.fold is not None
        
        self.label = self.compose_label()
        assert self.label is not None
        
        self.outputs = self.compose_outputs()
        assert self.outputs is not None
        
        self.post_compose()
    
    def post_compose(self):
        pass
    
    def compose_outputs(self):
        raise NotImplementedError
    
    def compose_index(self):
        raise NotImplementedError
    
    def compose_fold(self):
        raise NotImplementedError
    
    def compose_label(self):
        raise NotImplementedError
    
    @property
    def identifier(self):
        raise NotImplementedError
    
    @index_decorator
    def build_index(self, *args, **kwargs):
        raise NotImplementedError
    
    @fold_decorator
    def build_fold(self, *args, **kwargs):
        raise NotImplementedError
    
    @label_decorator
    def build_label(self, *args, **kwargs):
        raise NotImplementedError
    
    @data_decorator
    def build_data(self, modality, location, *args, **kwargs):
        raise NotImplementedError
