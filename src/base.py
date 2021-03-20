from functools import partial
from operator import itemgetter
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import pygraphviz as pgv
from loguru import logger
from mldb import ComputationGraph
from mldb import FileLockExistsException
from mldb import NodeWrapper
from mldb.backends import JsonBackend
from mldb.backends import NumpyBackend
from mldb.backends import PandasBackend
from mldb.backends import PickleBackend
from mldb.backends import PNGBackend
from mldb.backends import ScikitLearnBackend
from mldb.backends import YamlBackend

from src.keys import Key
from src.meta import BaseMeta
from src.utils.decorators import DecoratorBase
from src.utils.loaders import build_path
from src.utils.loaders import get_yaml_file_list
from src.utils.misc import NumpyEncoder
from src.utils.misc import randomised_order


__all__ = [
    "ExecutionGraph",
]


INDEX_FILES_SET = set(["index", "fold", "label", "target", "split"] + get_yaml_file_list("tasks", stem=True))


def is_index_key(key: Union[Key, str]) -> bool:
    return str(key) in INDEX_FILES_SET


def validate_meta(meta, name) -> BaseMeta:
    if isinstance(meta, BaseMeta):
        return meta
    elif isinstance(meta, (str, Path)):
        return BaseMeta(path=meta)
    elif isinstance(name, (str, Path)):
        return BaseMeta(path=name)

    logger.exception(f"Ambiguous metadata specification with {name=} and {meta=}")

    raise ValueError


class OutputGroup(object):
    def __init__(self, graph: "ExecutionGraph"):
        self.graph = graph

    def __repr__(self) -> str:
        graph_name = self.graph.name
        nodes = sorted(map(str, self.keys()))
        return f"{self.__class__.__name__}({graph_name=}, {nodes=})"

    def validate_key(self, key: Union[Key, str]) -> bool:
        return not is_index_key(key)

    @property
    def parent_group(self) -> "OutputGroup":
        return self.graph.parent.outputs

    def __getitem__(self, key: Union[Key, str]) -> NodeWrapper:
        assert self.validate_key(key)

        key = Key(key)

        if key in self.graph.nodes:
            return self.graph.nodes[key]

        if self.graph.parent is not None:
            return self.parent_group[key]

        logger.exception(f"Unable to find key '{key}' in graph - reached root.")

        raise KeyError

    def keys(self):
        yield from list(map(itemgetter(0), self.items()))

    def values(self):
        yield from list(map(itemgetter(0), self.items()))

    def items(self):
        keys = [key for key in self.graph.nodes.keys() if self.validate_key(key)]

        if len(keys) == 0:
            yield from self.parent_group.items()

        else:
            for key in keys:
                yield key, self.graph.nodes[key]


class IndexGroup(OutputGroup):
    def validate_key(self, key):
        return is_index_key(key)

    @property
    def parent_group(self) -> "IndexGroup":
        return self.graph.parent.index

    @property
    def index(self):
        return self["index"]

    @property
    def fold(self):
        return self["fold"]

    @property
    def split(self):
        return self["split"]


class ExecutionGraph(ComputationGraph):
    def __init__(self, name, parent=None, meta=None, default_backend="numpy"):
        super(ExecutionGraph, self).__init__(name=name)

        self.meta = validate_meta(meta=meta, name=name)

        self.parent: Optional["ExecutionGraph"] = parent

        self.fs_root: Path = build_path()

        self.add_backend("pickle", PickleBackend(self.fs_root))
        self.add_backend("pandas", PandasBackend(self.fs_root))
        self.add_backend("numpy", NumpyBackend(self.fs_root))
        self.add_backend("json", JsonBackend(self.fs_root, cls=NumpyEncoder))
        self.add_backend("sklearn", ScikitLearnBackend(self.fs_root))
        self.add_backend("png", PNGBackend(self.fs_root))
        self.add_backend("yaml", YamlBackend(self.fs_root))
        self.set_default_backend(default_backend)

        self.index = IndexGroup(graph=self)
        self.outputs = OutputGroup(graph=self)

        self.identifier.mkdir(exist_ok=True, parents=True)

    def relative_node_name(self, key: Union[Key, str]) -> Path:
        assert isinstance(key, (Key, str))
        return self.identifier / str(key)

    def absolute_node_name(self, key):
        return build_path(self.relative_node_name(key))

    # NODE CREATION/ACQUISITION

    def instantiate_orphan_node(
        self, func: Callable, backend: Optional[str] = None, kwargs: Optional[Dict[str, Any]] = None
    ) -> NodeWrapper:
        return self.make_node(name=None, func=func, backend=backend, kwargs=kwargs, cache=False)

    def instantiate_node(
        self,
        key: Union[Key, str],
        func: Callable,
        backend: Optional[str] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        force_add: bool = False,
    ) -> NodeWrapper:
        key = Key(key)
        if is_index_key(key):
            if backend != "pandas":
                logger.warning(f"Backend for node {key} in {self.identifier} is not pandas, setting.")
            backend = "pandas"
        if not force_add:
            assert key not in self.nodes
        name = self.absolute_node_name(key)
        return self.make_node(name=name, func=func, backend=backend, kwargs=kwargs, key=key)

    def get_or_create(
        self,
        key: Union[Key, str],
        func: Callable,
        backend: Optional[str] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> NodeWrapper:
        key = Key(key)
        if key in self.nodes:
            return self.nodes[key]
        return self.instantiate_node(key=key, func=func, backend=backend, kwargs=kwargs)

    def acquire_one(self, key: Union[Key, str], node: NodeWrapper) -> None:
        key = Key(key)
        if key in self.nodes:
            raise KeyError(f"{key} is already in {self.nodes.keys()} of {self}")
        self.nodes[key] = node

    def __getitem__(self, key: Union[Key, str]) -> NodeWrapper:
        if is_index_key(key):
            return self.index[key]
        return self.outputs[key]

    # BRANCHING

    def make_child(
        self, name: Union[Key, str], meta: Tuple[Path, str] = None, default_backend: Optional[str] = None
    ) -> "ExecutionGraph":
        return ExecutionGraph(
            name=name, parent=self, meta=meta, default_backend=default_backend or self.default_backend
        )

    def make_sibling(self, default_backend: Optional[str] = None):
        return ExecutionGraph(
            name=self.name, parent=self.parent, meta=self.meta, default_backend=default_backend or self.default_backend
        )

    def __truediv__(self, name: Union[Key, str]) -> "ExecutionGraph":
        return self.make_child(name=name)

    # EVALUATION

    @property
    def identifier(self) -> Path:
        if self.parent is None:
            return Path(self.name)
        return self.parent.identifier / self.name

    def get_ancestral_metadata(self, key: Union[Key, str]) -> Any:
        return _get_ancestral_meta(self, key)

    def dump_graph(self) -> None:
        dump_graph(graph=self, filename=self.absolute_node_name("graph.pdf"))

    def evaluate(self, force: bool = False) -> Dict[str, Any]:
        output = dict()
        for key in randomised_order(self.nodes.keys()):
            node = self.nodes[key]
            try:
                output[key] = node.evaluate()
            except FileLockExistsException:
                logger.warning(f"Skipping evaluation of {node.name} as it's already being computed.")
        return output

    # @staticmethod
    # def build_root():
    #     return ExecutionGraph(name="datasets")

    # @staticmethod
    # def zip_root():
    #     return ExecutionGraph("zips")


def dump_graph(graph, filename):
    nodes = dict()
    edges = []

    if isinstance(graph, NodeWrapper):
        consume_nodes(nodes, edges, graph)
    elif isinstance(graph, ExecutionGraph):
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
    try:
        G.layout("dot")
        filename.parent.mkdir(exist_ok=True, parents=True)
        G.draw(filename)
        G.close()
    except ValueError as ex:
        logger.exception(f"Unable to save dot file {filename}: {ex}")

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
            if isinstance(node.name, Path):
                name = f"{func_name} =>\n{node.name.stem}"
            else:
                name = func_name
            nodes[node_name] = f"{name}"
        return node_name

    add_node(ptr)
    for source_node in ptr.sources.values():
        source_name = add_node(source_node)
        edges.append((source_name, ptr.name))
        consume_nodes(nodes, edges, source_node)


def _get_ancestral_meta(graph: ExecutionGraph, key: Key):
    if graph.meta is None:
        logger.exception(TypeError(f'The key "{key}" cannot be found in "{graph}"'))
    if key in graph.meta:
        return graph.meta[key]
    if graph.parent is None:
        logger.exception(TypeError(f'The key "{key}" cannot be found in the ancestry of "{graph}"'))
    return _get_ancestral_meta(graph.parent, key)
