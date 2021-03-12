from numpy import concatenate

from src import Key

__all__ = [
    "modality_selector",
    "concatenate_features",
]


def do_select_feats(**nodes):
    keys = sorted(nodes.keys())
    return concatenate([nodes[key] for key in keys], axis=1)


def modality_selector(parent, modality="all", location="all"):
    locations_set = set(parent.get_ancestral_metadata("placements"))
    assert location in locations_set or location == "all"

    modality_set = set(parent.get_ancestral_metadata("modalities"))
    assert modality in modality_set or location == "all"

    root = parent / f"modality='{modality}'-location='{location}'"

    # Prepare a set of viable outputs
    valid_locations = set()
    for pos in parent.meta["sources"]:
        good_location = location == "all" or location == pos["placement"]
        good_modality = modality == "all" or modality == pos["modality"]
        if good_location and good_modality:
            valid_locations.update({Key((pos["modality"], pos["placement"]))})

    # Aggregate all relevant sources
    features = []
    for key, node in parent.outputs.items():
        if key in valid_locations:
            root.outputs.acquire_one(key, node)
            features.append((str(key), node))

    return root


def concatenate_features(parent):
    root = parent / f"concatenated"

    def concat(**nodes):
        keys = sorted(nodes.keys())
        return concatenate([nodes[key] for key in keys], axis=1)

    root.outputs.add_output(
        key="features", backend="none", func=concat, kwargs={str(kk): vv for kk, vv in parent.outputs.items()},
    )

    return root
