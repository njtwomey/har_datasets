import numpy as np
from src.selectors.base import SelectorBase

__all__ = [
    "select_feats",
]


def do_select_feats(key, **nodes):
    keys = sorted(nodes.keys())
    return np.concatenate([nodes[key] for key in keys], axis=1)


class select_feats(SelectorBase):
    def __init__(self, parent, name, **features):
        super(select_feats, self).__init__(name=name, parent=parent)

        self.outputs.add_output(
            key="features", backend="none", func=do_select_feats, kwargs=features,
        )

        self.evaluate_outputs()
