from pathlib import Path

from loguru import logger
from mldb import ComputationGraph
from mldb import FileLockExistsException
from mldb.backends import JsonBackend
from mldb.backends import NumpyBackend
from mldb.backends import PandasBackend
from mldb.backends import PickleBackend
from mldb.backends import PNGBackend
from mldb.backends import ScikitLearnBackend
from mldb.backends import VolatileBackend

from src.keys import Key
from src.meta import BaseMeta
from src.utils.loaders import build_path
from src.utils.loaders import get_yaml_file_list
from src.utils.misc import NumpyEncoder
from src.utils.misc import randomised_order


__all__ = [
    "BaseGraph",
]


def _get_ancestral_meta(graph, key):
    if graph.meta is None:
        logger.exception(TypeError(f'The key "{key}" cannot be found in "{graph}"'))
    if key in graph.meta:
        return graph.meta[key]
    if graph.parent is None:
        logger.exception(TypeError(f'The key "{key}" cannot be found in the ancestry of "{graph}"'))
    return _get_ancestral_meta(graph.parent, key)


def validate_ancestry(parent, sibling):
    if parent is not None and sibling is not None:
        logger.exception(
            ValueError(f"DAG cannot be specified consisting of both a parent and a sibling")
        )
    if parent is not None:
        return parent
    if sibling is not None:
        if not hasattr(sibling, "parent"):
            logger.exception(
                TypeError(
                    "The variable {sibling} is expected to have an attribute "
                    "called parent but does not."
                )
            )
        return sibling.parent
    return None


def validate_key_set_membership(key, key_set):
    return Key(key) in set(map(Key, key_set))
    # key = Key(key)
    # if len(key) != 1:
    #     return False
    # return key[0] in key_set


class ComputationalSet(object):
    def __init__(self, graph, parent=None, sibling=None, cat="outputs"):
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

        yaml_files = get_yaml_file_list("tasks", stem=True)
        self.index_keys = set(["index", "fold", "label", "target", "split"] + yaml_files)

        self.acquired = [self]

    def acquire_one(self, key, node):
        if key in self.output_dict:
            logger.exception(KeyError(f"{key} is not in {self.output_dict.keys()}"))
        self.output_dict[key] = node

    def acquire(self, other):
        for key, val in other.output_dict.items():
            self.acquire_one(key, val)

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
        key = Key(key)
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
        outputs = dict()
        for key in randomised_order(self.keys()):
            node = self[key]
            if not node.exists or force:
                if isinstance(node.backend, VolatileBackend):
                    continue

                try:
                    outputs[key] = node.evaluate()

                except FileLockExistsException as ex:
                    logger.warn(
                        f"The file {node.name} is currently being evaluated by a different process. "
                        f"Continuing to next available process: {ex}"
                    )
        return outputs

    def make_output(self, key, func, backend=None, kwargs=None):
        assert callable(func)

        key = Key(key)

        if kwargs is None:
            kwargs = dict()

        assert key not in self.output_dict

        node = self.graph.node(
            func=func,
            name=self.graph.build_path(key),
            backend=backend,
            kwargs=dict(key=key, **kwargs),
        )

        return node

    def append_output(self, key, node):
        key = Key(key)
        assert key not in self.output_dict
        self.output_dict[key] = node

    def add_output(self, key, func, backend=None, kwargs=None):
        node = self.make_output(key=key, func=func, backend=backend, kwargs=kwargs)

        self.append_output(key=key, node=node)

        return node


class IndexSet(ComputationalSet):
    def __init__(self, graph, parent):
        super(IndexSet, self).__init__(
            cat="index", graph=graph, parent=parent,
        )

    def validate_key(self, key):
        if not self.is_index_key(key):
            logger.exception(
                ValueError(
                    f"A non-index key was used for the index computation: {key} not in {self.index_keys}"
                )
            )

    def add_output(self, key, func, backend=None, kwargs=None):
        self.validate_key(key)
        return super(IndexSet, self).add_output(
            key=key,
            func=func,
            backend=backend or "pandas",  # Indexes default to pandas backend
            kwargs=kwargs,
        )

    def make_output(self, key, func, backend=None, kwargs=None):
        self.validate_key(key)
        return super(IndexSet, self).make_output(
            key=key, func=func, backend=backend or "pandas", kwargs=kwargs
        )

    @property
    def index(self):
        return self["index"]

    @property
    def fold(self):
        return self["fold"]


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

    def __init__(self, name, parent=None, meta=None, default_backend="numpy"):
        """
        
        Args:
            name:
            parent:
            meta:
            default_backend:
        """
        super(BaseGraph, self).__init__(name=name)

        if isinstance(meta, BaseMeta):
            self.meta = meta
        elif isinstance(meta, (str, Path)):
            self.meta = BaseMeta(path=meta)
        elif isinstance(name, (str, Path)):
            self.meta = BaseMeta(path=name)
        else:
            logger.exception(
                ValueError(f"Ambiguous metadata specification with name={name} and meta={meta}")
            )

        self.parent = parent

        self.fs_root = build_path()

        self.add_backend("pickle", PickleBackend(self.fs_root))
        self.add_backend("pandas", PandasBackend(self.fs_root))
        self.add_backend("numpy", NumpyBackend(self.fs_root))
        self.add_backend("json", JsonBackend(self.fs_root, cls=NumpyEncoder))
        self.add_backend("sklearn", ScikitLearnBackend(self.fs_root))
        self.add_backend("png", PNGBackend(self.fs_root))

        self.set_default_backend(default_backend)

        self.collections = ComputationalCollection(
            index=IndexSet(graph=self, parent=parent),
            outputs=ComputationalSet(graph=self, parent=parent),
        )

    def build_path(self, key):
        """
        
        Args:
            *args:

        Returns:

        """
        assert isinstance(key, Key)

        assert len(key) > 0
        if not isinstance(key[0], str):
            logger.exception(
                ValueError(
                    f"The argument for `build_path` must be strings, but got the type: {type(key[0])}"
                )
            )

        return build_path(self.identifier, str(key))

    def evaluate_outputs(self):
        """
        
        Returns:

        """

        outputs = dict()
        for key, output in self.collections.items():
            outputs[key] = output.evaluate_outputs()
        return outputs

    @property
    def index(self):
        return self.collections["index"]

    @property
    def outputs(self):
        return self.collections["outputs"]

    @property
    def identifier(self):
        if self.parent is None:
            return Path(self.name)
        return self.parent.identifier / self.name

    def get_ancestral_metadata(self, key):
        return _get_ancestral_meta(self, key)
