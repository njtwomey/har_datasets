from src.keys import Key

__all__ = [
    'make_key_subset_of', 'accel_filt', 'gyro_filt', 'take_all'
]


def make_key_subset_of(key_set):
    key_set = Key(key_set)

    def key_subset_of(key):
        return set(key_set.key).issubset(set(Key(key).key))

    return key_subset_of


def accel_filt():
    return make_key_subset_of('accel')


def gyro_filt():
    return make_key_subset_of('gyro')


def take_all():
    return lambda *args, **kwargs: True
