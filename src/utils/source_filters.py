from src.base import make_key

__all__ = [
    'make_key_subset_of', 'accel_filt', 'gyro_filt', 'take_all'
]


def make_key_subset_of(key_set):
    key_set = set(make_key(key_set))
    
    def key_subset_of(key):
        return key_set.issubset(make_key(key))
    
    return key_subset_of


def accel_filt():
    return make_key_subset_of('accel')


def gyro_filt():
    return make_key_subset_of('gyro')


def take_all():
    return lambda *args, **kwargs: True
