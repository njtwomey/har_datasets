from functools import partial
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import ItemsView
from typing import Iterable
from typing import KeysView
from typing import Optional
from typing import Tuple
from typing import Union
from typing import ValuesView

import pygraphviz as pgv
from loguru import logger
from mldb import ComputationGraph
from mldb import FileLockExistsException
from mldb import NodeWrapper
from mldb.backends import Backend
from mldb.backends import JsonBackend
from mldb.backends import NumpyBackend
from mldb.backends import PandasBackend
from mldb.backends import PickleBackend
from mldb.backends import PNGBackend
from mldb.backends import ScikitLearnBackend
from mldb.backends import VolatileBackend

from src.keys import Key
from src.meta import BaseMeta
from src.utils.decorators import DecoratorBase
from src.utils.loaders import build_path
from src.utils.loaders import get_yaml_file_list
from src.utils.misc import NumpyEncoder
from src.utils.misc import randomised_order


__all__ = [
    "BaseGraph",
]


INDEX_FILES_SET = set(["index", "fold", "label", "target", "split"] + get_yaml_file_list("tasks", stem=True))


def is_index_key(key: Union[Key, str]) -> bool:
    assert isinstance(key, (Key, str))
    return str(key) in INDEX_FILES_SET


class BaseGraph(ComputationGraph):
    def __init__(self, name, parent=None, meta=None, default_backend="numpy"):
        super(BaseGraph, self).__init__(name=name)

        if isinstance(meta, BaseMeta):
            self.meta: BaseMeta = meta
        elif isinstance(meta, (str, Path)):
            self.meta: BaseMeta = BaseMeta(path=meta)
        elif isinstance(name, (str, Path)):
            self.meta: BaseMeta = BaseMeta(path=name)
        else:
            logger.exception(ValueError(f"Ambiguous metadata specification with name={name} and meta={meta}"))

        self.parent: Optional["BaseGraph"] = parent

        self.fs_root: Path = build_path()

        self.add_backend("pickle", PickleBackend(self.fs_root))
        self.add_backend("pandas", PandasBackend(self.fs_root))
        self.add_backend("numpy", NumpyBackend(self.fs_root))
        self.add_backend("json", JsonBackend(self.fs_root, cls=NumpyEncoder))
        self.add_backend("sklearn", ScikitLearnBackend(self.fs_root))
        self.add_backend("png", PNGBackend(self.fs_root))
        self.set_default_backend(default_backend)

        self.index: IndexNodeGroup = IndexNodeGroup(graph=self, default_backend="pandas")
        self.outputs: OutputNodeGroup = OutputNodeGroup(graph=self)

    def build_path(self, key: Union[Key, str]) -> Path:
        assert isinstance(key, Key) and len(key)
        return build_path(self.identifier, str(key))

    def evaluate(self, force: bool = False) -> Dict[Key, Any]:
        return self.outputs.evaluate(force=force)

    @property
    def identifier(self) -> Path:
        if self.parent is None:
            return Path(self.name)
        return self.parent.identifier / self.name

    def get_ancestral_metadata(self, key: Union[Key, str]) -> Any:
        return _get_ancestral_meta(self, key)

    def __truediv__(self, name: Union[Key, str]) -> "BaseGraph":
        return self.make_child(name=name)

    def __getitem__(self, key: Union[Key, str]) -> NodeWrapper:
        if is_index_key(key):
            return self.index[key]
        return self.outputs[key]

    def make_child(
        self, name: Union[Key, str], meta: Tuple[Path, str] = None, default_backend: str = "numpy"
    ) -> "BaseGraph":
        return BaseGraph(name=name, parent=self, meta=meta, default_backend=default_backend)

    def dump_graph(self) -> None:
        dump_graph(graph=self, filename=build_path(self.identifier, "graph.pdf"))

    @staticmethod
    def build_root():
        return BaseGraph("datasets")

    @staticmethod
    def zip_root():
        return BaseGraph("zips")


class OutputNodeGroup(object):
    def __init__(self, graph: BaseGraph, default_backend: Optional[str] = None):
        self.graph: BaseGraph = graph
        self.default_backend: Optional[str] = default_backend
        self.output_dict: Dict[Key, NodeWrapper] = dict()

    def validate_key(self, key: Union[Key, str]) -> None:
        assert not is_index_key(key), f"An index key was used for {self}: {key} in {INDEX_FILES_SET}"

    def acquire_one(self, key: Union[Key, str], node: NodeWrapper) -> None:
        if key in self.output_dict:
            raise KeyError(f"{key} is already in {self.output_dict.keys()} of {self}")
        self.output_dict[key] = node

    def acquire(self, other: "OutputNodeGroup") -> None:
        for key, val in other.output_dict.items():
            self.acquire_one(key, val)

    @property
    def parent_outputs(self):
        return self.graph.parent.outputs

    @property
    def active_comp_set(self) -> Dict[Key, NodeWrapper]:
        if len(self.output_dict) == 0:
            return self.parent_outputs
        return self.output_dict

    def __getitem__(self, key: Union[Key, str]) -> NodeWrapper:
        key = Key(key)
        try:
            return self.output_dict[key]
        except KeyError:
            if self.graph.parent is None:
                logger.exception(ValueError(f"Unable to find key '{key}' in graph - reached root."))
            return self.parent_outputs[key]

    def keys(self) -> KeysView[Key]:
        return self.active_comp_set.keys()

    def values(self) -> ValuesView[NodeWrapper]:
        return self.active_comp_set.values()

    def items(self) -> ItemsView[Key, NodeWrapper]:
        return self.active_comp_set.items()

    def __len__(self) -> int:
        return len(self.active_comp_set)

    def __contains__(self, item: Union[Key, str]) -> bool:
        return item in self.active_comp_set

    def __repr__(self):
        return f"<{self.__class__.__name__} outputs={self.keys()}/>"

    def evaluate(self, force: bool = False) -> Dict[Key, Any]:
        outputs = dict()
        for key in randomised_order(self.keys()):
            node = self[key]
            if not node.exists or force:
                if isinstance(node.backend, VolatileBackend):
                    continue
                try:
                    outputs[key] = node.evaluate()
                except FileLockExistsException as ex:
                    logger.warning(
                        f"The file {node.name} is currently being evaluated by a different process. "
                        f"Continuing to next available process: {ex}"
                    )
        return outputs

    def instantiate_node(
        self,
        key: Union[Key, str],
        func: Callable,
        backend: Optional[Backend] = None,
        kwargs: Optional[Dict[Any, Any]] = None,
    ) -> NodeWrapper:
        self.validate_key(key)
        assert callable(func)
        key = Key(key)
        if kwargs is None:
            kwargs = dict()
        assert key not in self.output_dict
        node = self.graph.node(func=func, name=self.graph.build_path(key), backend=backend, kwargs=dict(**kwargs),)
        return node

    def append_node(self, key: Union[Key, str], node: NodeWrapper) -> None:
        self.validate_key(key)
        key = Key(key)
        assert key not in self.output_dict
        self.output_dict[key] = node

    def create(
        self, key: Union[Key, str], func: Callable, backend: Optional[str] = None, kwargs: Dict[str, Any] = None
    ) -> NodeWrapper:
        self.validate_key(key)
        node = self.instantiate_node(key=key, func=func, backend=backend or self.default_backend, kwargs=kwargs)
        self.append_node(key=key, node=node)
        return node

    def get_or_create(
        self, key: Union[Key, str], func: Callable, backend: Optional[str] = None, kwargs: Dict[str, Any] = None
    ) -> NodeWrapper:
        key = Key(key)
        if key in self.output_dict:
            return self.output_dict[key]
        return self.create(key=key, func=func, backend=backend, kwargs=kwargs)


class IndexNodeGroup(OutputNodeGroup):
    def __init__(self, graph: BaseGraph, default_backend: Optional[str] = "pandas"):
        super(IndexNodeGroup, self).__init__(graph=graph, default_backend=default_backend)

    @property
    def parent_outputs(self):
        return self.graph.parent.index

    @property
    def active_comp_set(self):
        if len(self.output_dict) == 0:
            return self.graph.parent.index
        return self.output_dict

    def validate_key(self, key):
        assert is_index_key(key), f"A non-index key was used for {self}: {key} not in {INDEX_FILES_SET}"

    @property
    def index(self):
        return self["index"]

    @property
    def fold(self):
        return self["fold"]

    @property
    def split(self):
        return self["split"]


def dump_graph(graph, filename):
    nodes = dict()
    edges = []

    if isinstance(graph, NodeWrapper):
        consume_nodes(nodes, edges, graph)
    elif isinstance(graph, BaseGraph):
        for _, node in graph.outputs.items():
            consume_nodes(nodes, edges, node)
    else:
        raise TypeError

    nodes = {str(kk): vv for kk, vv in nodes.items()}
    edges = list(map(lambda rr: list(map(str, rr)), edges))

    G = pgv.AGraph(directed=True, strict=True, rankdir="LR")
    for node_id, node_name in nodes.items():
        G.add_node(node_id, label=node_name)
    G.add_edges_from(edges)
    G.layout("dot")
    filename.parent.mkdir(exist_ok=True, parents=True)
    G.draw(filename)
    G.close()

    return nodes, edges


def consume_nodes(nodes, edges, ptr):
    def add_node(node):
        node_name = node.name
        func = node.func
        if isinstance(func, partial):
            func = node.func.func.__self__.func
        elif isinstance(node.func, DecoratorBase):
            func = func.func
        func_name = func.__name__
        if node_name not in nodes:
            nodes[node_name] = f"{func_name} =>\n{node.name.stem}"
        return node_name

    add_node(ptr)
    for source_node in ptr.sources.values():
        source_name = add_node(source_node)
        edges.append((source_name, ptr.name))
        consume_nodes(nodes, edges, source_node)


def _get_ancestral_meta(graph: BaseGraph, key: Key):
    if graph.meta is None:
        logger.exception(TypeError(f'The key "{key}" cannot be found in "{graph}"'))
    if key in graph.meta:
        return graph.meta[key]
    if graph.parent is None:
        logger.exception(TypeError(f'The key "{key}" cannot be found in the ancestry of "{graph}"'))
    return _get_ancestral_meta(graph.parent, key)
