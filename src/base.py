from pathlib import Path

from mldb import ComputationGraph, FileLockExistsException
from mldb.backends import (
    PickleBackend, JsonBackend, PandasBackend, NumpyBackend,
    ScikitLearnBackend, PNGBackend, VolatileBackend
)

from src.meta import BaseMeta
from src.utils.misc import NumpyEncoder, randomised_order
from src.utils.logger import get_logger
from src.utils.loaders import build_path, get_yaml_file_list

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
        logger.exception(TypeError(
            f'The key "{key}" cannot be found in "{graph}"'
        ))
    if key in graph.meta:
        return graph.meta[key]
    if graph.parent is None:
        logger.exception(TypeError(
            f'The key "{key}" cannot be found in the ancestry of "{graph}"'
        ))
    return _get_ancestral_meta(graph.parent, key)


def validate_ancestry(parent, sibling):
    if parent is not None and sibling is not None:
        logger.exception(ValueError(
            f'DAG cannot be specified consisting of both a parent and a sibling'
        ))
    if parent is not None:
        return parent
    if sibling is not None:
        if not hasattr(sibling, 'parent'):
            logger.exception(TypeError(
                'The variable {sibling} is expected to have an attribute '
                'called parent but does not.'
            ))
        return sibling.parent
    return None


def validate_key_set_membership(key, key_set):
    key = make_key(key)
    if len(key) != 1:
        return False
    return key[0] in key_set


class ComputationalSet(object):
    def __init__(self, graph, parent=None, sibling=None, cat='outputs'):
        """
        
        Args:
            graph:
            parent:
            cat:
        """
        self.cat = cat
        self.graph = graph

        self.parent = validate_ancestry(parent, sibling)
        self.output_dict = dict()

        yaml_files = get_yaml_file_list('tasks', stem=True)
        self.index_keys = set(
            ['index', 'fold', 'label', 'target', 'split'] + yaml_files
        )

        self.acquired = [self]

    def acquire(self, other):
        for key, val in other.output_dict.items():
            if key in self.output_dict:
                logger.exception(KeyError(
                    f'{key} is not in {self.output_dict.keys()}'
                ))
            self.output_dict[key] = val
        logger.info(f'Added keys to CompSet: {len(self.output_dict)}, {self.output_dict.keys()}')

    @property
    def active_comp_set(self):
        if len(self.output_dict) == 0:
            return self.parent.collections[self.cat]
        return self.output_dict

    def keys(self):
        return self.active_comp_set.keys()

    def items(self):
        return self.active_comp_set.items()

    def __len__(self):
        return len(self.active_comp_set)

    def __iter__(self):
        yield from self.active_comp_set

    def __contains__(self, item):
        return item in self.active_comp_set

    def __getitem__(self, key):
        key = make_key(key)
        try:
            return self.output_dict[key]
        except KeyError:
            assert self.parent is not None
            return self.parent.collections[self.cat][key]

    def __repr__(self):
        return f"<{self.__class__.__name__} outputs={self.output_dict}/>"

    def is_index_key(self, key):
        return validate_key_set_membership(key, self.index_keys)

    def evaluate_outputs(self, force=False):
        for key in randomised_order(self.keys()):
            node = self[key]
            if not node.exists or force:
                if isinstance(node.backend, VolatileBackend):
                    logger.info(f'Not evaluating {key} by default since it is of a volatile backend: {node.name}')
                    continue

                try:
                    logger.info(f'Calculating {node.name}')
                    node.evaluate()

                except FileLockExistsException as ex:
                    logger.warn(
                        f'The file {node.name} is currently being evaluated by a different process. '
                        f'Continuing to next available process: {ex}'
                    )

            else:
                logger.info(f'Loading {node.name}')

    def make_output(self, key, func, backend=None, **kwargs):
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
        key = make_key(key)
        assert key not in self.output_dict
        self.output_dict[key] = node
        logger.info(f'Node {node.name} added to outputs: {self.output_dict.keys()}.')

    def add_output(self, key, func, backend=None, **kwargs):
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
        super(IndexSet, self).__init__(
            cat='index',
            graph=graph,
            parent=parent,
        )

    def validate_key(self, key):
        if not self.is_index_key(key):
            logger.exception(ValueError(
                f"A non-index key was used for the index computation: {key} not in {self.index_keys}"
            ))

    def add_output(self, key, func, backend=None, **kwargs):
        self.validate_key(key)
        return super(IndexSet, self).add_output(
            key=key,
            func=func,
            backend=backend or 'pandas',  # Indexes default to pandas backend
            **kwargs
        )

    def make_output(self, key, func, backend=None, **kwargs):
        self.validate_key(key)
        return super(IndexSet, self).make_output(
            key=key, func=func, backend=backend or 'pandas', **kwargs
        )

    @property
    def index(self):
        return self['index']

    @property
    def fold(self):
        return self['fold']


class ComputationalCollection(object):
    def __init__(self, **kwargs):
        self.comp_dict = dict()
        for kk, vv in kwargs.items():
            self.append(kk, vv)

    def append(self, name, collection):
        assert name not in self.comp_dict
        self.comp_dict[name] = collection
        setattr(self, name, collection)

    def __getitem__(self, key):
        return self.comp_dict[key]

    def items(self):
        return self.comp_dict.items()

    def keys(self):
        return self.comp_dict.keys()

    def values(self):
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

        if isinstance(meta, BaseMeta):
            self.meta = meta

        elif isinstance(meta, (str, Path)):
            self.meta = BaseMeta(path=meta)

        elif isinstance(name, (str, Path)):
            self.meta = BaseMeta(path=name)

        else:
            logger.exception(ValueError(
                f'Ambiguous metadata specification with name={name} and meta={meta}'
            ))

        self.parent = parent

        self.fs_root = build_path()

        self.add_backend('pickle', PickleBackend(self.fs_root))
        self.add_backend('pandas', PandasBackend(self.fs_root))
        self.add_backend('numpy', NumpyBackend(self.fs_root))
        self.add_backend('json', JsonBackend(self.fs_root, cls=NumpyEncoder))
        self.add_backend('sklearn', ScikitLearnBackend(self.fs_root))
        self.add_backend('png', PNGBackend(self.fs_root))

        self.set_default_backend(default_backend)

        self.collections = ComputationalCollection(
            index=IndexSet(
                graph=self, parent=parent
            ),
            outputs=ComputationalSet(
                graph=self, parent=parent
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
            logger.exception(ValueError(
                f'The argument for `build_path` must be strings, but got the type: {type(args[0])}'
            ))

        return build_path(self.identifier, '-'.join(args))

    def evaluate_outputs(self):
        """
        
        Returns:

        """
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
        """
        
        Returns:

        """
        if self.parent is None:
            return Path(self.name)
        return self.parent.identifier / self.name

    def get_ancestral_metadata(self, key):
        """
        
        Args:
            key:

        Returns:

        """
        return _get_ancestral_meta(self, key)
