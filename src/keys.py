__all__ = ["Key"]

from pathlib import Path
from typing import Union


class Key(object):
    def __init__(self, key):
        self.key = validate_key(key)

    def __len__(self) -> int:
        return len(self.key)

    def __str__(self) -> str:
        return self.key

    def __repr__(self) -> str:
        key = self.key
        return f"Key({key=})"

    def __eq__(self, other: Union["Key", Path, str]) -> bool:
        if isinstance(other, Path):
            return self.key == other.name
        if isinstance(other, Key):
            return self.key == other.key
        if isinstance(other, str):
            return self.key == other
        raise NotImplementedError

    def __hash__(self) -> int:
        return hash(self.key)

    def __contains__(self, key) -> bool:
        validate_key(key)
        return key in self.key

    def __lt__(self, other) -> bool:
        return self.key < other.key


def validate_key(key: Union[Path, Key, str]):
    if isinstance(key, Path):
        return key.name
    if isinstance(key, Key):
        return key.key
    if isinstance(key, str):
        return key
    raise ValueError(f"Unsupported key type: expected str or Key, got {type(key)} (value: {key})")
