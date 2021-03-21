from functools import lru_cache
from functools import partial
from operator import itemgetter
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

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
from mldb.backends import YamlBackend

from src.functional.common import node_itemgetter
from src.keys import Key
from src.meta import BaseMeta
from src.utils.decorators import DecoratorBase
from src.utils.loaders import build_path
from src.utils.loaders import get_yaml_file_list
from src.utils.misc import NumpyEncoder
from src.utils.misc import randomised_order


__all__ = ["ExecutionGraph", "get_ancestral_metadata"]


INDEX_FILES_SET = set(
    get_yaml_file_list("indices", stem=True)
    + get_yaml_file_list("tasks", stem=True)
    + get_yaml_file_list("splits", stem=True)
)

DATA_ROOT: Path = build_path()

BACKEND_DICT = dict(
    none=VolatileBackend(),
    pickle=PickleBackend(DATA_ROOT),
    pandas=PandasBackend(DATA_ROOT),
    numpy=NumpyBackend(DATA_ROOT),
    json=JsonBackend(DATA_ROOT, cls=NumpyEncoder),
    sklearn=ScikitLearnBackend(DATA_ROOT),
    png=PNGBackend(DATA_ROOT),
    yaml=YamlBackend(DATA_ROOT),
)


@lru_cache(2 ** 16)
def is_index_key(key: Optional[Union[Key, str]]) -> bool:
    if key is None:
        return False
    if isinstance(key, Key):
        key = key.key
    assert isinstance(key, str)
    return key in INDEX_FILES_SET


def validate_meta(meta, name) -> BaseMeta:
    if isinstance(meta, BaseMeta):
        return meta
    elif isinstance(meta, (str, Path)):
        return BaseMeta(path=meta)
    elif isinstance(name, (str, Path)):
        return BaseMeta(path=name)

    logger.exception(f"Ambiguous metadata specification with {name=} and {meta=}")

    raise ValueError


def validate_backend(backend: Optional[str], key: Optional[Union[str, Key]] = None) -> Backend:
    if is_index_key(key):
        if backend != "pandas":
            logger.warning(f"Backend for node {key} is not pandas - setting value to 'pandas'.")
        backend = "pandas"

    else:
        if backend is None:
            backend = "none"

    if backend not in BACKEND_DICT:
        logger.exception(f"Backend ({backend}) not in known list ({sorted(BACKEND_DICT.keys())})")
        raise KeyError

    return BACKEND_DICT[backend]


def relative_node_name(identifier: Path, key: Union[Key, str]) -> Path:
    assert isinstance(key, (Key, str))
    return identifier / str(key)


def absolute_node_name(identifier: Path, key: Union[Key, str]) -> Path:
    return DATA_ROOT / relative_node_name(identifier=identifier, key=key)


class NodeGroup(object):
    def __init__(self, graph: "ExecutionGraph"):
        self.graph = graph

    def __repr__(self) -> str:
        graph_name = self.graph.name
        nodes = sorted(map(str, self.keys()))
        return f"{self.__class__.__name__}({graph_name=}, {nodes=})"

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
        yield from map(itemgetter(0), self.items())

    def values(self):
        yield from map(itemgetter(1), self.items())

    def items(self):
        keys = [key for key in self.graph.nodes.keys() if self.validate_key(key)]

        if len(keys) == 0:
            yield from self.parent_group.items()

        else:
            for key in keys:
                yield key, self.graph.nodes[key]

    def validate_key(self, key: Union[Key, str]) -> bool:
        raise NotImplementedError

    @property
    def parent_group(self) -> "NodeGroup":
        raise NotImplementedError


class OutputGroup(NodeGroup):
    def validate_key(self, key: Union[Key, str]) -> bool:
        return not is_index_key(key)

    @property
    def parent_group(self) -> "OutputGroup":
        return self.graph.parent.outputs


class IndexGroup(NodeGroup):
    def validate_key(self, key):
        return is_index_key(key)

    @property
    def parent_group(self) -> "IndexGroup":
        return self.graph.parent.index

    def get_split_series(self, split, fold):
        return self.graph.instantiate_node(
            key=f"{split=}-{fold=}", func=node_itemgetter(fold), backend="pandas", args=self[split]
        )

    @property
    def index(self):
        return self["index"]

    @property
    def har(self):
        return self["har"]

    @property
    def predefined(self):
        return self["predefined"]

    @property
    def loso(self):
        return self["loso"]

    @property
    def deployable(self):
        return self["deployable"]


class ExecutionGraph(ComputationGraph):
    def __init__(self, name, parent=None, meta=None):
        super(ExecutionGraph, self).__init__(name=name)
        self.meta = validate_meta(meta=meta, name=name)
        self.parent: Optional["ExecutionGraph"] = parent
        self.index = IndexGroup(graph=self)
        self.outputs = OutputGroup(graph=self)

    # NODE CREATION/ACQUISITION

    def instantiate_orphan_node(
        self,
        func: Callable,
        backend: Optional[str] = None,
        args: Optional[Union[Any, List[Any], Tuple[Any]]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> NodeWrapper:
        backend = validate_backend(backend)
        return self.make_node(name=None, func=func, backend=backend, args=args, kwargs=kwargs, cache=False)

    def instantiate_node(
        self,
        key: Union[Key, str],
        func: Callable,
        backend: Optional[str] = None,
        args: Optional[Union[Any, List[Any], Tuple[Any]]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        force_add: bool = False,
    ) -> NodeWrapper:
        key = Key(key)
        if not force_add:
            assert key not in self.nodes
        name = absolute_node_name(identifier=self.identifier, key=key)
        backend = validate_backend(backend, key)
        return self.make_node(name=name, key=key, func=func, backend=backend, args=args, kwargs=kwargs)

    def get_or_create(
        self,
        key: Union[Key, str],
        func: Callable,
        backend: Optional[str] = None,
        args: Optional[Union[Any, Tuple[Any]]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> NodeWrapper:
        key = Key(key)
        if key in self.nodes:
            return self.nodes[key]
        return self.instantiate_node(key=key, func=func, backend=backend, args=args, kwargs=kwargs)

    def acquire_node(self, node: NodeWrapper, key: Optional[Union[Key, str]] = None) -> None:
        if key is None:
            raise NotImplementedError
        key = Key(key)
        if key in self.nodes:
            raise KeyError(f"Cannot acquire {key} since a node of this name's already in {self.nodes.keys()} of {self}")
        self.nodes[key] = node

    def __getitem__(self, key: Union[Key, str]) -> NodeWrapper:
        if is_index_key(key):
            return self.index[key]
        return self.outputs[key]

    # BRANCHING

    def make_child(self, name: Union[Key, str], meta: Tuple[Path, str] = None) -> "ExecutionGraph":
        return ExecutionGraph(name=name, parent=self, meta=meta)

    def make_sibling(self):
        # TODO: should this inherit index?

        logger.warning("Making siblings not tested - be wary!")

        return ExecutionGraph(name=self.name, parent=self.parent, meta=self.meta)

    def __truediv__(self, name: Union[Key, str]) -> "ExecutionGraph":
        return self.make_child(name=name)

    # EVALUATION

    @property
    def identifier(self) -> Path:
        if self.parent is None:
            return Path(self.name)
        return self.parent.identifier / self.name

    def dump_graph(self) -> None:
        dump_graph(graph=self, filename=absolute_node_name(identifier=self.identifier, key="graph.pdf"))

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


def get_all_sources(node: NodeWrapper):
    sources = []

    def resolve(nn):
        if isinstance(nn, NodeWrapper):
            sources.append(nn)
        elif isinstance(nn, (list, tuple)):
            for ni in nn:
                resolve(ni)
        elif isinstance(nn, dict):
            for ni in nn.values():
                resolve(ni)

    resolve(node.args)
    resolve(node.kwargs)

    return sources


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

    for source_node in get_all_sources(ptr):
        source_name = add_node(source_node)
        edges.append((source_name, ptr.name))
        consume_nodes(nodes, edges, source_node)


def get_ancestral_metadata(graph: Union[NodeWrapper, ExecutionGraph], key: str):
    if isinstance(graph, NodeWrapper):
        graph = graph.graph
    if graph.meta is None:
        logger.exception(f'The key "{key}" cannot be found in "{graph}"')
        raise KeyError
    if key in graph.meta:
        return graph.meta[key]
    if graph.parent is None:
        logger.exception(f'The key "{key}" cannot be found in the ancestry of "{graph}"')
        raise ValueError
    return get_ancestral_metadata(graph.parent, key)
