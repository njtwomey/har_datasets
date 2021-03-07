import numpy as np

from src.selectors.base import SelectorBase

__all__ = [
    "concatenate_features",
]


def do_select_feats(key, **nodes):
    keys = sorted(nodes.keys())
    return np.concatenate([nodes[key] for key in keys], axis=1)


class concatenate_features(SelectorBase):
    def __init__(self, parent, **features):
        super(concatenate_features, self).__init__(name="concatenated", parent=parent)

        self.outputs.add_output(
            key="features", backend="none", func=do_select_feats, kwargs=features
        )

        self.evaluate_outputs()
