__all__ = ["Key"]


def validate_key(key):
    if isinstance(key, Key):
        return key.key
    if key is None:
        key = tuple()
    if isinstance(key, str):
        key = (key,)
    assert isinstance(key, tuple)
    return key


class Key(object):
    def __init__(self, args):
        if isinstance(args, Key):
            self.key = args.key
        self.key = validate_key(args)

    def __len__(self):
        return len(self.key)

    def __str__(self):
        return "-".join(self.key)

    def __getitem__(self, item):
        return self.key[item]

    def __repr__(self):
        return f"<Key key={self.key}>"

    def __eq__(self, other):
        return self.key == other.key

    def __hash__(self):
        return hash(self.key)

    def __add__(self, other):
        if isinstance(other, str):
            return Key(self.key + (other,))
        elif isinstance(other, tuple):
            return Key(self.key + other)
        elif isinstance(other, Key):
            return Key(self.key + other.key)
        raise TypeError

    def __contains__(self, item):
        return item in self.key
