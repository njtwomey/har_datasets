from .utils import load_metadata, build_path, check_activities

__all__ = [
    'DatasetMeta', 'BaseMeta', 'ActivityMeta', 'LocationMeta', 'ModalityMeta', 'DatasetMeta',
    'FeatureMeta', 'TransformerMeta', 'VisualisationMeta', 'PipelineMeta', 'ModelMeta',
]


class BaseMeta(object):
    def __init__(self, name, yaml_file, *args, **kwargs):
        values = load_metadata(yaml_file)
        assert name in values, f'The function "{name}" is not in the set {{{values}}} found in {yaml_file}'
        self.name = name
        self.meta = values[name]
        if self.meta is None:
            self.meta = dict()
    
    def __getitem__(self, item):
        assert item in self.meta, f'{item} not found in {self.__class__.__name__}'
        return self.meta[item]
    
    def __contains__(self, item):
        return item in self.meta
    
    def keys(self):
        return self.meta.keys()
    
    def values(self):
        return self.meta.values()
    
    def items(self):
        return self.meta.items()
    
    def __repr__(self):
        return f'<{self.name} {self.meta.__repr__()}>'
    
    def insert(self, key, value):
        assert key not in self.meta
        self.meta[key] = value


"""
Non-functional metadata
"""


class ActivityMeta(BaseMeta):
    def __init__(self, name, *args, **kwargs):
        super(ActivityMeta, self).__init__(
            name=name, yaml_file='activities.yaml'
        )


class LocationMeta(BaseMeta):
    def __init__(self, name, *args, **kwargs):
        super(LocationMeta, self).__init__(
            name=name, yaml_file='locations.yaml'
        )


class ModalityMeta(BaseMeta):
    def __init__(self, name, *args, **kwargs):
        super(ModalityMeta, self).__init__(
            name=name, yaml_file='modalities.yaml'
        )


class DatasetMeta(BaseMeta):
    def __init__(self, name, *args, **kwargs):
        super(DatasetMeta, self).__init__(
            name=name, yaml_file='datasets.yaml'
        )
        
        assert 'fs' in self.meta
        
        self.inv_act_lookup = check_activities(self.meta['activities'])
    
    @property
    def fs(self):
        return float(self.meta['fs'])
    
    @property
    def zip_path(self):
        return build_path('data', 'zip', self.name)
    
    @property
    def build_path(self):
        return build_path('data', 'build', self.name)


class FeatureMeta(BaseMeta):
    def __init__(self, name, *args, **kwargs):
        super(FeatureMeta, self).__init__(
            name=name, yaml_file='features.yaml'
        )


class TransformerMeta(BaseMeta):
    def __init__(self, name, *args, **kwargs):
        super(TransformerMeta, self).__init__(
            name=name, yaml_file='transformers.yaml'
        )


class VisualisationMeta(BaseMeta):
    def __init__(self, name, *args, **kwargs):
        super(VisualisationMeta, self).__init__(
            name=name, yaml_file='visualisations.yaml',
        )


class PipelineMeta(BaseMeta):
    def __init__(self, name, *args, **kwargs):
        super(PipelineMeta, self).__init__(
            name=name, yaml_file='pipelines.yaml'
        )


class ModelMeta(BaseMeta):
    def __init__(self, name, *args, **kwargs):
        super(ModelMeta, self).__init__(
            name=name, yaml_file='models.yaml'
        )
