from os.path import join

from mldb import ComputationGraph, PickleBackend, JsonBackend, PandasBackend, NumpyBackend

from src.meta import BaseMeta
from src.utils.misc import NumpyEncoder, randomised_order
from src.utils.logger import get_logger
from src.utils.loaders import build_path

logger = get_logger(__name__)

__all__ = [
    'BaseGraph',
]


# TODO/FIXME: may be worth defining a key type

def make_key(key):
    """
    
    Args:
        key:

    Returns:

    """
    if key is None:
        key = tuple()
    if isinstance(key, str):
        key = (key,)
    assert isinstance(key, tuple)
    return key


def _get_ancestral_meta(graph, key):
    """
    
    Args:
        graph:
        key:

    Returns:

    """
    if graph.meta is None:
        logger.exception(f'The key "{key}" cannot be found in "{graph}"')
        raise TypeError
    if key in graph.meta:
        return graph.meta[key]
    if graph.parent is None:
        logger.exception(f'The key "{key}" cannot be found in the ancestry of "{graph}"')
        raise TypeError
    return _get_ancestral_meta(graph.parent, key)


class ComputationalSet(object):
    def __init__(self, graph, parent, cat='outputs'):
        """
        
        Args:
            graph:
            parent:
            cat:
        """
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
    def active_comp_set(self):
        """
        
        Returns:

        """
        if len(self.output_dict) == 0:
            return self.parent.collections[self.cat]
        return self.output_dict
    
    def keys(self):
        """
        
        Returns:

        """
        return self.active_comp_set.keys()
    
    def items(self):
        """
        
        Returns:

        """
        return self.active_comp_set.items()
    
    def __len__(self):
        """
        
        Returns:

        """
        return len(self.active_comp_set)
    
    def __iter__(self):
        """
        
        Returns:

        """
        yield from self.active_comp_set
    
    def __contains__(self, item):
        """
        
        Args:
            item:

        Returns:

        """
        return item in self.active_comp_set
    
    def __getitem__(self, key):
        """
        
        Args:
            key:

        Returns:

        """
        key = make_key(key)
        try:
            return self.output_dict[key]
        except KeyError:
            assert self.parent is not None
            return self.parent.collections[self.cat][key]
    
    def __repr__(self):
        """
        
        Returns:

        """
        return f"<{self.__class__.__name__} outputs={self.output_dict}/>"
    
    def is_index_key(self, key):
        """
        
        Args:
            key:

        Returns:

        """
        key = make_key(key)
        if len(key) != 1:
            return False
        return key[0] in self.index_keys
    
    def evaluate_outputs(self, force=False):
        """
        
        Args:
            force:

        Returns:

        """
        for key in randomised_order(self.keys()):
            node = self[key]
            if not node.exists or force:
                logger.info(f'Calculating {node.name}')
                node.evaluate()
            else:
                logger.info(f'Loading {node.name}')
    
    def make_output(self, key, func, backend=None, **kwargs):
        """
        
        Args:
            key:
            func:
            backend:
            **kwargs:

        Returns:

        """
        assert callable(func)
        
        key = make_key(key)
        
        assert key not in self.output_dict
        
        node = self.graph.node(
            name=self.graph.build_path(*key),
            func=func,
            backend=backend,
            key=key,
            **kwargs,
        )
        
        logger.info(f'Created node {node.name}; backend: {backend}')
        
        return node
    
    def append_output(self, key, node):
        """
        
        Args:
            key:
            node:

        Returns:

        """
        
        key = make_key(key)
        assert key not in self.output_dict
        self.output_dict[key] = node
        logger.info(f'Node {node.name} added to outputs: {self.output_dict.keys()}.')
    
    def add_output(self, key, func, backend=None, **kwargs):
        """
        
        Args:
            key:
            func:
            backend:
            **kwargs:

        Returns:

        """
        node = self.make_output(
            key=key,
            func=func,
            backend=backend,
            **kwargs
        )
        
        self.append_output(key=key, node=node)
        
        return node


class IndexSet(ComputationalSet):
    def __init__(self, graph, parent):
        """
        
        Args:
            graph:
            parent:
        """
        super(IndexSet, self).__init__(
            cat='index',
            graph=graph,
            parent=parent,
        )
        
    def validate_key(self, key):
        if not self.is_index_key(key):
            logger.exception(f"A non-index key was used for the index computation: {key} not in {self.index_keys}")
            raise ValueError
    
    def add_output(self, key, func, backend=None, **kwargs):
        """
        
        Args:
            key:
            func:
            backend:
            **kwargs:

        Returns:

        """
        self.validate_key(key)
        return super(IndexSet, self).add_output(
            key=key,
            func=func,
            backend=backend or 'pandas',  # Indexes default to pandas backend
            **kwargs
        )
    
    def make_output(self, key, func, backend=None, **kwargs):
        """
        
        Args:
            key:
            func:
            backend:
            **kwargs:

        Returns:

        """
        self.validate_key(key)
        return super(IndexSet, self).make_output(
            key=key, func=func, backend=backend or 'pandas', **kwargs
        )
    
    @property
    def index(self):
        """
        
        Returns:

        """
        return self['index']
    
    @property
    def label(self):
        """
        
        Returns:

        """
        return self['label']
    
    @property
    def fold(self):
        """
        
        Returns:

        """
        return self['fold']


class ComputationalCollection(object):
    def __init__(self, **kwargs):
        """
        
        Args:
            **kwargs:
        """
        self.comp_dict = dict()
        for kk, vv in kwargs.items():
            self.append(kk, vv)
    
    def append(self, name, node):
        """
        
        Args:
            name:
            node:

        Returns:

        """
        self.comp_dict[name] = node
        setattr(self, name, node)
    
    def __getitem__(self, key):
        """
        
        Args:
            key:

        Returns:

        """
        return self.comp_dict[key]
    
    def items(self):
        """
        
        Returns:

        """
        return self.comp_dict.items()
    
    def keys(self):
        """
        
        Returns:

        """
        return self.comp_dict.keys()
    
    def values(self):
        """
        
        Returns:

        """
        return self.comp_dict.values()


class BaseGraph(ComputationGraph):
    """
    A simple computational graph that is meant only to define backends and load metadata
    """
    
    def __init__(self, name, parent=None, meta=None, default_backend='numpy'):
        """
        
        Args:
            name:
            parent:
            meta:
            default_backend:
        """
        super(BaseGraph, self).__init__(
            name=name,
        )
        
        if not isinstance(meta, BaseMeta):
            logger.exception(f"The metadata variable does not derive from `BaseMeta` but is of type {type(meta)}")
            raise TypeError
        
        self.meta = meta
        
        self.parent = parent
        
        self.fs_root = build_path()
        
        self.add_backend('pickle', PickleBackend(self.fs_root))
        self.add_backend('pandas', PandasBackend(self.fs_root))
        self.add_backend('numpy', NumpyBackend(self.fs_root))
        self.add_backend('json', JsonBackend(self.fs_root, cls=NumpyEncoder))
        
        self.set_default_backend(default_backend)
        
        self.collections = ComputationalCollection(
            index=IndexSet(
                graph=self,
                parent=parent
            ),
            outputs=ComputationalSet(
                graph=self,
                parent=parent
            ),
        )
    
    def build_path(self, *args):
        """
        
        Args:
            *args:

        Returns:

        """
        assert len(args) > 0
        if not isinstance(args[0], str):
            logger.exception(
                f'The argument for `build_path` must be strings, but got the type: {type(args[0])}'
            )
            raise ValueError
        
        return build_path(join(self.identifier, '-'.join(args)))
    
    def evaluate_outputs(self):
        """
        
        Returns:

        """
        for key, output in self.collections.items():
            output.evaluate_outputs()
    
    @property
    def index(self):
        """
        
        Returns:

        """
        return self.collections['index']
    
    @property
    def outputs(self):
        """
        
        Returns:

        """
        return self.collections['outputs']
    
    @property
    def identifier(self):
        """
        
        Returns:

        """
        if self.parent is None:
            return self.name
        return join(
            self.parent.identifier,
            self.name,
        )
    
    def get_ancestral_metadata(self, key):
        """
        
        Args:
            key:

        Returns:

        """
        return _get_ancestral_meta(self, key)
