from numpy import concatenate

from src.base import ExecutionGraph
from src.keys import Key

__all__ = [
    "source_selector",
    "concatenate_features",
]


def do_select_feats(**nodes):
    keys = sorted(nodes.keys())
    return concatenate([nodes[key] for key in keys], axis=1)


def source_selector(parent, modality="all", location="all"):
    locations_set = set(parent.get_ancestral_metadata("locations"))
    assert location in locations_set or location == "all", f"Location {location} not in {locations_set}"

    modality_set = set(parent.get_ancestral_metadata("modalities"))
    assert modality in modality_set or modality == "all", f"Modality {modality} not in {modality_set}"

    loc, mod = location, modality
    root = parent / f"{loc=}-{mod=}"

    # Prepare a set of viable outputs
    valid_locations = set()
    for pair in parent.meta["sources"]:
        loc, mod = pair["loc"], pair["mod"]
        good_location = location == "all" or pair["loc"] == location
        good_modality = modality == "all" or pair["mod"] == modality
        if good_location and good_modality:
            valid_locations.update({Key(f"{loc=}-{mod=}")})

    # Aggregate all relevant sources
    for key, node in parent.outputs.items():
        if key in valid_locations:
            root.acquire_one(key, node)

    return root


def concatenate_nodes(parent, **kwargs):
    def concat(**nodes):
        keys = sorted(nodes.keys())
        return concatenate([nodes[key] for key in keys], axis=1)

    return parent.instantiate_orphan_node(backend="none", func=concat, kwargs=kwargs,)


def concatenate_features(parent: ExecutionGraph):
    def concat(**nodes):
        keys = sorted(nodes.keys())
        return concatenate([nodes[key] for key in keys], axis=1)

    return parent.instantiate_orphan_node(
        backend="none", func=concat, kwargs={str(kk): vv for kk, vv in parent.outputs.items()},
    )
