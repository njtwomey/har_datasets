import numpy as np
from src.selectors.base import SelectorBase

__all__ = [
    'select_feats',
]


def do_select_feats(key, **nodes):
    keys = sorted(nodes.keys())
    return np.concatenate(
        [nodes[key] for key in keys], axis=1
    )


class select_feats(SelectorBase):
    def __init__(self, parent, source_filter, source_name):
        assert source_filter and source_name
        
        super(select_feats, self).__init__(
            name=source_name, parent=parent
        )
        
        nodes = dict()
        for key, node in parent.outputs.items():
            if source_filter(key):
                nodes[key] = node
        
        self.outputs.add_output(
            key=source_name, backend='none',
            func=do_select_feats,
            data=parent,
            **nodes
        )
