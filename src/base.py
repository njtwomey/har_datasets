from collections import defaultdict
from os.path import join

from mldb import ComputationGraph, PickleBackend, VolatileBackend, JsonBackend
from .utils.backends import PandasBackend, NumpyBackend

from .utils import build_path

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
    
    def __contains__(self, item):
        return item in self.output_dict
    
    def __getitem__(self, key):
        key = key_check(key)
        if key not in self.output_dict:
            raise ValueError(f'The key "{key}" not in {list(self.output_dict.keys())}')
        return self.output_dict[key]
    
    def evaluate_outputs(self, force=False):
        for key, node in self.items():
            # print(key, node)
            if not node.exists or force:
                node.evaluate()
    
    def make_output(self, key, func, sources=None, backend=None, **kwargs):
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
        
        return node
    
    def append_output(self, key, node):
        key = key_check(key)
        assert key not in self.output_dict
        self.output_dict[key] = node
    
    def add_output(self, key, func, sources=None, backend=None, **kwargs):
        node = self.make_output(
            key=key,
            func=func,
            sources=sources,
            backend=backend,
            **kwargs
        )
        
        self.append_output(key=key, node=node)
        
        return node


class IndexSet(ComputationalSet):
    def __init__(self, graph, parent):
        super(IndexSet, self).__init__(graph=graph)
        self.parent = parent
    
    def add_output(self, key, func, sources=None, backend=None, **kwargs):
        assert self.is_index_key(key)
        return super(IndexSet, self).add_output(
            key=key,
            func=func,
            sources=sources,
            backend=backend or 'pandas',
            **kwargs
        )
    
    def make_output(self, key, func, sources=None, backend=None, **kwargs):
        return super(IndexSet, self).make_output(
            key=key, func=func, sources=sources, backend=backend or 'pandas', **kwargs
        )
    
    def clone_all_from_parent(self, parent, **kwargs):
        for key in self.index_keys:
            self.clone_from_parent(key=key, parent=parent)
    
    def clone_from_parent(self, key, parent):
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
        )
    
    def __getitem__(self, key):
        """
        Overwrite to automatically inherit index values from the ancestry

        :param key:
        :return:
        """
        try:
            return super(IndexSet, self).__getitem__(key)
        except ValueError:
            assert self.parent is not None
            return self.parent.index[key]
    
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
    
    def __getitem__(self, key):
        return self.items[key]


class BaseGraph(ComputationGraph):
    """
    A simple computational graph that is meant only to define backends and load metadata
    """
    
    def __init__(self, name, parent=None, meta=None, default_backend='numpy'):
        super(BaseGraph, self).__init__(
            name=name.lower(),
        )
        
        self.parent = parent
        
        self.fs_root = build_path('data')
        
        self.add_backend('pickle', PickleBackend(self.fs_root))
        self.add_backend('json', JsonBackend(self.fs_root))
        self.add_backend('pandas', PandasBackend(self.fs_root))
        self.add_backend('numpy', NumpyBackend(self.fs_root))
        self.add_backend('none', VolatileBackend())
        
        self.set_default_backend(default_backend)
        
        self.collections = ComputationalCollection(
            index=IndexSet(graph=self, parent=parent),
            outputs=ComputationalSet(graph=self),
        )
        
        self.meta = meta
    
    def build_path(self, *args):
        assert len(args) > 0
        assert isinstance(args[0], str), f'The argument for `build_path` must be strings, but is {type(args[0])}'
        path = build_path('data', 'build', self.identifier, '-'.join(args))
        return path
    
    def evaluate_outputs(self):
        # warnings.warn(len(self.index), f'The index graph for {self} is empty')
        self.index.evaluate_outputs()
        
        # warnings.warn(len(self.outputs), f'The output graph for {self} is empty')
        self.outputs.evaluate_outputs()
    
    @property
    def index(self):
        return self.collections['index']
    
    @property
    def outputs(self):
        return self.collections['outputs']
    
    @property
    def identifier(self):
        if self.parent is None:
            return self.name
        return join(
            self.parent.identifier,
            self.name,
        )
    
    def get_ancestral_metadata(self, key):
        return _get_ancestral_meta(self, key)
