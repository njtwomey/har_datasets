from os.path import join, splitext, basename, curdir
from os import environ

from collections import defaultdict, namedtuple

from mldb import ComputationGraph, PickleBackend, VolatileBackend

from .utils import check_activities, check_modalities, check_locations, unzip_data, load_yaml, build_path
from .utils.decorators import label_decorator, index_decorator, data_decorator, fold_decorator

__all__ = [
    'DatasetMeta', 'BaseGraph', 'FuncKeyMap',
    'BaseMeta', 'ActivityMeta', 'LocationMeta', 'ModalityMeta', 'DatasetMeta', 'Window', 'FeatureMeta',
    'TransformerMeta', 'RepresentationMeta', 'ModelMeta'
]


class BaseMeta(object):
    def __init__(self, name, yaml_file, *args, **kwargs):
        values = load_yaml(yaml_file)
        assert name in values
        self.name = name
        self.meta = values[name]
    
    def __getitem__(self, item):
        assert item in self.meta, f'{item} not found in {self.__class__.__name__}'
        return self.meta[item]
    
    def add_category(self, key, value):
        assert key not in self.meta
        self.meta[key] = value


class FuncKeyMap(namedtuple('FuncKeyMap', ('in_key', 'out_key', 'func'))):
    def __new__(cls, in_key, out_key, func):
        if in_key is None:
            in_key = tuple()
        in_key = in_key if isinstance(in_key, tuple) else (in_key,)
        assert out_key is not None
        out_key = out_key if isinstance(out_key, tuple) else (out_key,)
        out_key = in_key + out_key
        return super(FuncKeyMap, cls).__new__(cls, in_key, out_key, func)


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


class Window(object):
    def __init__(self, win_len=None, win_inc=None, *args, **kwargs):
        self.win_len = win_len
        self.win_inc = win_inc
    
    def dir_name(self):
        if self.win_len is None:
            return curdir
        return join(
            f'win_len={self.win_len}s',
            f'win_inc={self.win_inc}s',
        )


class FeatureMeta(BaseMeta):
    def __init__(self, name, *args, **kwargs):
        super(FeatureMeta, self).__init__(
            name=name, yaml_file='features.yaml'
        )
    
    @property
    def windowed(self):
        return self.meta.get('windowed', False)
    
    @property
    def window(self):
        return Window(
            **self.meta['win_spec']
        )


class TransformerMeta(BaseMeta):
    def __init__(self, name, *args, **kwargs):
        super(TransformerMeta, self).__init__(
            name=name, yaml_file='transformers.yaml'
        )


class RepresentationMeta(BaseMeta):
    def __init__(self, name, *args, **kwargs):
        super(RepresentationMeta, self).__init__(
            name=name, yaml_file='representation.yaml'
        )


class ModelMeta(BaseMeta):
    def __init__(self, name, *args, **kwargs):
        super(ModelMeta, self).__init__(
            name=name, yaml_file='model.yaml'
        )


def walk_dict(dict_var, path=None):
    if path is None:
        path = tuple()
    for kk, vv in dict_var.items():
        next_path = path
        if isinstance(kk, tuple):
            next_path += kk
        else:
            next_path += (kk,)
        if isinstance(vv, (dict, defaultdict)):
            yield from walk_dict(vv, next_path)
        else:
            yield next_path, vv


class BaseGraph(ComputationGraph):
    """
    A simple computational graph that is meant only to define backends and load metadata
    """
    
    def __init__(self, name):
        super(BaseGraph, self).__init__(
            name=name, default_backend='fs'
        )
        
        self.fs_root = build_path('data')
        
        self.add_backend('fs', PickleBackend(self.fs_root))
        self.add_backend('none', VolatileBackend())
        
        self.label = None
        self.fold = None
        self.index = None
        
        self.outputs = None
        
        self.output_list = None
        self.extra_args = None
        
        self.composed = False

    def get_index(self, key):
        return dict(
            label=self.label,
            fold=self.fold,
            index=self.index
        )[key]
    
    def add_extra_kwargs(self, **kwargs):
        if self.extra_args is None:
            self.extra_args = dict()
        self.extra_args.update(**kwargs)
    
    def add_output(self, in_key, out_key, func):
        assert out_key is not None
        assert func is not None
        if self.output_list is None:
            self.output_list = []
        self.output_list.append(FuncKeyMap(
            in_key=in_key, out_key=out_key, func=func
        ))
        
    # def add_outputs(self, *args):
    #     ass
    #     for vv in args:
    #         assert isinstance(vv, FuncKeyMap)
    #     self.output_list.extend(args)
    
    def build_path(self, *args):
        assert isinstance(args[0], str), 'The argument for `build_path` must be strings'
        return build_path('data', 'build', self.identifier, '-'.join(args))
    
    def iter_outputs(self):
        assert isinstance(self.outputs, (dict, defaultdict))
        for kk, vv in walk_dict(self.outputs):
            yield kk, vv
    
    def compose_check(self):
        if not self.composed:
            self.compose()
            self.composed = True
    
    def evaluate_all(self, if_exists=False):
        self.compose_check()
        super(BaseGraph, self).evaluate_all(if_exists=if_exists)
    
    def evaluate_outputs(self, if_exists=False):
        self.compose_check()
        for node in self.outputs.values():
            if (not node.exists) or (node.exists and if_exists):
                node.evaluate()
    
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

    def compose_index(self):
        return self.compose_meta('index')

    def compose_fold(self):
        return self.compose_meta('fold')

    def compose_label(self):
        return self.compose_meta('label')
    
    def compose_outputs(self):
        outputs = dict()
        for in_key, out_key, func in self.output_list:
            outputs[out_key] = self.make_node(in_key, out_key, func)
        return outputs
    
    def compose_meta(self, name):
        raise NotImplementedError
    
    def make_node(self, in_key, out_key, func):
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
    def build_data(self, key, *args, **kwargs):
        raise NotImplementedError
