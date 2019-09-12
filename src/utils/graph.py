__all__ = [
    'get_ancestral_meta'
]


def get_ancestral_meta(node, key):
    if key not in node.meta:
        assert hasattr(node, 'parent')
        return get_ancestral_meta(node.parent, key)
    return node.meta[key]
