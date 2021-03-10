from numpy import concatenate

__all__ = [
    "modality_selector",
]


def do_select_feats(**nodes):
    keys = sorted(nodes.keys())
    return concatenate([nodes[key] for key in keys], axis=1)


def modality_selector(parent, view="all", location="all"):
    locations_set = set(parent.get_ancestral_metadata("placements"))
    assert location in locations_set or location == "all"

    modality_set = set(parent.get_ancestral_metadata("modalities"))
    assert view in modality_set or location == "all"

    root = parent / f"view={view}-loc={location}"

    # Aggregate all relevant sources
    features = []
    for key, node in parent.outputs.items():
        has_view = view == "all" or view in key
        has_location = location == "all" or location in key
        if has_view and has_location:
            features.append((str(key), node))

    # Concatenate the features
    root.outputs.add_output(
        key="features", backend="none", func=do_select_feats, kwargs=dict(features)
    )

    return root
