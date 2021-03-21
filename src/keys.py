__all__ = ["Key"]

from pathlib import Path


def validate_key(key):
    if isinstance(key, Path):
        return key.name
    if isinstance(key, Key):
        return key.key
    if isinstance(key, str):
        return key
    raise ValueError(f"Unsupported key type: expected str or Key, got {type(key)} (value: {key})")


class Key(object):
    def __init__(self, key):
        self.key = validate_key(key)

    def __len__(self):
        return len(self.key)

    def __str__(self):
        return self.key

    def __getitem__(self, item):
        raise NotImplementedError

    def __repr__(self):
        key = self.key
        return f"Key({key=})"

    def __eq__(self, other):
        if isinstance(other, Path):
            return self.key == other.name
        if isinstance(other, Key):
            return self.key == other.key
        if isinstance(other, str):
            return self.key == other
        raise NotImplementedError

    def __hash__(self):
        return hash(self.key)

    def __add__(self, other):
        raise NotImplementedError

    def __contains__(self, key):
        validate_key(key)
        return key in self.key

    def __lt__(self, other):
        return self.key < other.key
