from os import environ
from os.path import join
from .utils import load_metadata, get_logger

logger = get_logger(__name__)

__all__ = [
    'DatasetMeta', 'BaseMeta', 'HARMeta', 'PlacementMeta', 'ModalityMeta', 'DatasetMeta',
]


def check_yaml(name, values):
    value_set = load_metadata(name)
    for value in values:
        if value not in value_set:
            logger.exception(ValueError(
                f'{value} is not yet in `{name}.yaml`'
            ))
    if isinstance(values, dict):
        return {vv: kk for kk, vv in values.items()}


def check_activity(values):
    return check_yaml(join('tasks', 'har.yaml'), values)


def check_localisation(values):
    return check_yaml(join('tasks', 'localisation.yaml'), values)


def check_modality(values):
    return check_yaml('modality.yaml', values)


def check_placement(values):
    return check_yaml('placement.yaml', values)


class BaseMeta(object):
    def __init__(self, name, yaml_file=None, is_sub_cat=False, *args, **kwargs):
        self.name = name
        self.meta = dict()
        
        if yaml_file:
            logger.info(f'Loading metadata file {yaml_file} for {name}.')
            
            try:
                meta = load_metadata(yaml_file)
                
                if is_sub_cat:
                    if name not in meta:
                        logger.exception(KeyError(
                            f'The function "{name}" is not in the set {{{meta}}} that is listed in {yaml_file}'
                        ))
                    meta = meta[name]
                
                if meta is None:
                    logger.info(f'The content metadata module {name} is empty. Assigning empty dict')
                else:
                    assert isinstance(meta, dict)
                    self.meta = meta
            
            except FileNotFoundError:
                logger.warn(f'The metadata file for {name} was not found.')
    
    def __getitem__(self, item):
        if item not in self.meta:
            logger.exception(KeyError(
                f'{item} not found in {self.__class__.__name__}'
            ))
        return self.meta[item]
    
    def __contains__(self, item):
        return item in self.meta
    
    def __repr__(self):
        return f'<{self.name} {self.meta.__repr__()}>'
    
    def keys(self):
        return self.meta.keys()
    
    def values(self):
        return self.meta.values()
    
    def items(self):
        return self.meta.items()
    
    def insert(self, key, value):
        assert key not in self.meta
        logger.info(f'The key "{key}" is being manually inserted to metadata {self.name}')
        self.meta[key] = value


"""
Non-functional metadata
"""


class HARMeta(BaseMeta):
    def __init__(self, name, *args, **kwargs):
        super(HARMeta, self).__init__(
            name=name, yaml_file=join('tasks', 'har.yaml'), *args, **kwargs
        )


class LocalisationMeta(BaseMeta):
    def __init__(self, name, *args, **kwargs):
        super(LocalisationMeta, self).__init__(
            name=name, yaml_file=join('tasks', 'localisation.yaml'), *args, **kwargs
        )


class PlacementMeta(BaseMeta):
    def __init__(self, name, *args, **kwargs):
        super(PlacementMeta, self).__init__(
            name=name, yaml_file='placement.yaml', *args, **kwargs
        )


class ModalityMeta(BaseMeta):
    def __init__(self, name, *args, **kwargs):
        super(ModalityMeta, self).__init__(
            name=name, yaml_file='modality.yaml', *args, **kwargs
        )


class DatasetMeta(BaseMeta):
    def __init__(self, name, *args, **kwargs):
        super(DatasetMeta, self).__init__(
            name=name, yaml_file=join('datasets', f'{name}.yaml'), *args, **kwargs
        )
        
        if 'fs' not in self.meta:
            logger.exception(KeyError(
                f'The metadata for {name} does not contain the required key "fs"'
            ))
        
        self.inv_lookup = dict()
        for task_name in self.meta['tasks'].keys():
            self.inv_lookup[task_name] = check_activity(
                self.meta['tasks'][task_name]['target_transform']
            )
    
    @property
    def fs(self):
        return float(self.meta['fs'])
    
    @property
    def zip_path(self):
        return join(environ['ZIP_ROOT'], self.name)
