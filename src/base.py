from collections import defaultdict, namedtuple

from mldb import ComputationGraph, PickleBackend, VolatileBackend

from .utils import build_path
from .meta import *

__all__ = [
    'BaseGraph',
]


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


def key_check(key):
    if key is None:
        key = tuple()
    if isinstance(key, str):
        key = (key,)
    assert isinstance(key, tuple)
    return key


def _get_ancestral_meta(graph, key):
    assert graph.meta is not None, f'The key "{key}" cannot be found in "{graph}"'
    if key in graph.meta:
        return graph.meta[key]
    assert graph.parent is not None, f'The key "{key}" cannot be found in the ancestry of "{graph}"'
    return _get_ancestral_meta(graph.parent, key)


class ComputationalSet(object):
    def __init__(self, graph):
        self.graph = graph
        self.output_dict = dict()
        
        self.index_keys = {
            'index',
            'label',
            'fold'
        }
    
    def is_index_key(self, key):
        if isinstance(key, tuple):
            assert len(key) == 1
            key = key[0]
        return key in self.index_keys
    
    def keys(self):
        return self.output_dict.keys()
    
    def items(self):
        return self.output_dict.items()
    
    def __len__(self):
        return len(self.output_dict)
    
    def __iter__(self):
        yield from self.output_dict
    
    def __repr__(self):
        return f"<{self.__class__.__name__} outputs={self.output_dict}/>"
    
    def __getitem__(self, key):
        key = key_check(key)
        assert key in self.output_dict, f'The key "{key}" not in {list(self.output_dict.keys())}'
        return self.output_dict[key]
    
    def evaluate_all(self, force=False):
        for node in self.node_list:
            if (not node.exists) or (node.exists and force):
                node.evaluate()
    
    def add_output(self, key, func, sources=None, backend=None, **kwargs):
        assert callable(func)
        
        key = key_check(key)
        
        assert key not in self.output_dict
        
        node = self.graph.node(
            node_name=self.graph.build_path(*key),
            func=func,
            sources=sources,
            backend=backend,
            kwargs=dict(key=key, **kwargs),
        )
        
        self.output_dict[key] = node
        
        return node


class IndexSet(ComputationalSet):
    def __init__(self, graph):
        super(IndexSet, self).__init__(graph=graph)
    
    def add_output(self, key, func, sources=None, backend=None, **kwargs):
        assert self.is_index_key(key)
        return super(IndexSet, self).add_output(
            key=key,
            func=func,
            sources=sources,
            backend=backend,
            **kwargs
        )
    
    def evaluate_all(self, force=False):
        return super(IndexSet, self).evaluate_all(force=force)
    
    def clone_from_parent(self, key, parent, **kwargs):
        assert self.is_index_key(key)
        
        def identity(key, index, data):
            return data
        
        return self.add_output(
            key=key,
            func=identity,
            sources=dict(
                index=parent.index.index,
                data=parent.index[key],
            ),
            **kwargs
        )
    
    @property
    def index(self):
        return self['index']
    
    @property
    def label(self):
        return self['label']
    
    @property
    def fold(self):
        return self['fold']


class ComputationalCollection(object):
    def __init__(self, **kwargs):
        self.items = kwargs
        for kk, vv in kwargs.items():
            setattr(self, kk, vv)
    
    def iter_outputs(self):
        yield from self.items.items()
    
    @property
    def is_composed(self):
        return all(vv.is_composed for kk, vv in self.iter_outputs())
    
    def compose(self):
        for kk, vv in self.iter_outputs():
            vv.compose()
    
    def evaluate_all(self):
        for kk, vv in self.iter_outputs():
            vv.evaluate_all()
    
    def __getitem__(self, key):
        return self.items[key]


class BaseGraph(ComputationGraph):
    """
    A simple computational graph that is meant only to define backends and load metadata
    """
    
    def __init__(self, name, parent=None, default_backend='fs'):
        super(BaseGraph, self).__init__(
            default_backend=default_backend,
            name=name,
        )
        
        self.parent = parent
        
        self.fs_root = build_path('data')
        
        self.add_backend('fs', PickleBackend(self.fs_root))
        self.add_backend('none', VolatileBackend())
        
        self.collections = ComputationalCollection(
            index=IndexSet(self),
            outputs=ComputationalSet(self),
        )
        
        self.meta = None
        self.extra_args = None
    
    @property
    def index(self):
        return self.collections['index']
    
    @property
    def outputs(self):
        return self.collections['outputs']
    
    @property
    def identifier(self):
        raise NotImplementedError
    
    def build_path(self, *args):
        assert isinstance(args[0], str), 'The argument for `build_path` must be strings'
        path = build_path('data', 'build', self.identifier, '-'.join(args))
        return path
    
    def evaluate_all(self, if_exists=False):
        assert len(self.index), f'The index graph for {self} is empty'
        assert len(self.outputs), f'The output graph for {self} is empty'
        return super(BaseGraph, self).evaluate_all(if_exists=if_exists)
    
    def get_ancestral_metadata(self, key):
        return _get_ancestral_meta(self, key)
