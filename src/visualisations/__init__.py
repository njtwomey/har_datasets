from .umap_embedding import umap_embedding

__all__ = [
    'load_visualisation',
    'umap_embedding',
]


def load_visualisation(*args, **kwargs):
    visualisations = {kk: globals()[kk] for kk in __all__}
    assert args[0] in visualisations
    return visualisations[args[0]](*args[1:], **kwargs)
