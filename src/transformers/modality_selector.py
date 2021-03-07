from numpy import concatenate

from src.transformers.base import TransformerBase

__all__ = [
    "modality_selector",
]


def do_select_feats(key, **nodes):
    keys = sorted(nodes.keys())
    feats = concatenate([nodes[key] for key in keys], axis=1)
    print("feats:", feats.shape)
    return feats


class modality_selector(TransformerBase):
    def __init__(self, parent, modality="all", location="all"):
        super(modality_selector, self).__init__(name=self.__class__.__name__, parent=parent)

        self.locations_set = set(self.get_ancestral_metadata("placements"))
        assert location in self.locations_set or location == "all"
        self.view_name = modality

        self.modality_set = set(self.get_ancestral_metadata("modalities"))
        assert modality in self.modality_set or location == "all"
        self.location_name = location

        # Aggregate all relevant sources
        features = []
        for key, node in parent.outputs.items():
            has_view = modality == "all" or modality in key
            has_location = location == "all" or location in key
            if has_view and has_location:
                features.append((str(key), node))

        # Concatenate the features
        self.outputs.add_output(
            key="features", backend="none", func=do_select_feats, kwargs=dict(features)
        )

    @property
    def identifier(self):
        return self.parent.identifier / f"view={self.view_name}-loc={self.location_name}"
