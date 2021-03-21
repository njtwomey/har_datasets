from loguru import logger
from numpy import concatenate

from src.base import get_ancestral_metadata
from src.keys import Key

__all__ = [
    "source_selector",
]


def do_select_feats(**nodes):
    keys = sorted(nodes.keys())
    return concatenate([nodes[key] for key in keys], axis=1)


def source_selector(parent, modality="all", location="all"):
    locations_set = set(get_ancestral_metadata(parent, "locations"))
    assert location in locations_set or location == "all", f"Location {location} not in {locations_set}"

    modality_set = set(get_ancestral_metadata(parent, "modalities"))
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
    selected = 0
    for key, node in parent.outputs.items():
        if key in valid_locations:
            selected += 1
            root.acquire_node(key=key, node=node)

    if not selected:
        logger.exception(f"No wearable keys found in {sorted(parent.outputs.keys())}")
        raise KeyError

    return root
