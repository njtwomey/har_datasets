from collections import defaultdict
from os.path import join

from mldb import ComputationGraph, PickleBackend, VolatileBackend, JsonBackend
from .utils.backends import PandasBackend, NumpyBackend

from .utils import build_path, get_logger, randomised_order

logger = get_logger(__name__)

__all__ = [
    'BaseGraph',
]


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
    def __init__(self, graph, parent, cat='outputs'):
        self.cat = cat
        self.graph = graph
        self.parent = parent
        self.output_dict = dict()
        
        self.index_keys = {
            'index',
            'label',
            'fold'
        }
    
    @property
    def parent_comp_set(self):
        return self.parent.collections[self.cat]
    
    def keys(self):
        if len(self.output_dict) == 0:
            return self.parent_comp_set.keys()
        return self.output_dict.keys()
    
    def items(self):
        if len(self.output_dict) == 0:
            return self.parent_comp_set.items()
        return self.output_dict.items()
    
    def __len__(self):
        if len(self.output_dict) == 0:
            return len(self.parent_comp_set)
        return len(self.output_dict)
    
    def __iter__(self):
        if len(self.output_dict) == 0:
            yield from self.parent_comp_set
        yield from self.output_dict
    
    def __contains__(self, item):
        if len(self.output_dict) == 0:
            return item in self.parent_comp_set
        return item in self.output_dict
    
    def __getitem__(self, key):
        key = key_check(key)
        try:
            return self.output_dict[key]
        except KeyError:
            assert self.parent is not None
            return self.parent_comp_set[key]
    
    def __repr__(self):
        return f"<{self.__class__.__name__} outputs={self.output_dict}/>"
    
    def is_index_key(self, key):
        key = key_check(key)
        if len(key) != 1:
            return False
        return key[0] in self.index_keys
    
    def evaluate_outputs(self, force=False):
        for key in randomised_order(self.keys()):
            node = self[key]
            logger.info(f'Evaluating the node {node.name}')
            if not node.exists or force:
                logger.info(f'Calculating {node.name}')
                node.evaluate()
            else:
                logger.info(f'Loading {node.name}')
    
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
        
        logger.info(f'Created node {node.name}; backend: {backend}')
        
        return node
    
    def append_output(self, key, node):
        key = key_check(key)
        assert key not in self.output_dict
        self.output_dict[key] = node
        logger.info(f'Node {node.name} added to outputs.')
    
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
        super(IndexSet, self).__init__(
            cat='index',
            graph=graph,
            parent=parent,
        )
    
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
        self._items = kwargs
        for kk, vv in kwargs.items():
            setattr(self, kk, vv)
    
    def iter_outputs(self):
        yield from self._items.items()
    
    def __getitem__(self, key):
        return self._items[key]
    
    def items(self):
        return self._items.items()
    
    def keys(self):
        return self._items.keys()
    
    def values(self):
        return self._items.values()


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
        self.add_backend('pandas', PandasBackend(self.fs_root))
        self.add_backend('numpy', NumpyBackend(self.fs_root))
        self.add_backend('json', JsonBackend(self.fs_root))
        self.add_backend('none', VolatileBackend())
        
        self.set_default_backend(default_backend)
        
        self.collections = ComputationalCollection(
            index=IndexSet(graph=self, parent=parent),
            outputs=ComputationalSet(graph=self, parent=parent),
        )
        
        self.meta = meta
    
    def build_path(self, *args):
        assert len(args) > 0
        assert isinstance(args[0], str), f'The argument for `build_path` must be strings, but is {type(args[0])}'
        path = build_path('data', 'build', self.identifier, '-'.join(args))
        return path
    
    def evaluate_outputs(self):
        for key, output in self.collections.items():
            output.evaluate_outputs()
    
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
