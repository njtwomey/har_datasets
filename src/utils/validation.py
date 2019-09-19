from numpy import isfinite

from .loaders import load_metadata

__all__ = [
    'check_yaml', 'check_modalities', 'check_locations', 'check_activities',
    'get_validator',
    'all_finite',
]

"""
YAML checkers
"""


def check_yaml(name, values):
    value_set = load_metadata(name)
    for value in values:
        if value not in value_set:
            raise ValueError(f'{value} is not yet in `{name}.yaml`')
    if isinstance(values, dict):
        return {vv: kk for kk, vv in values.items()}


def check_activities(values):
    return check_yaml('activities.yaml', values)


def check_locations(values):
    return check_yaml('locations.yaml', values)


def check_modalities(values):
    return check_yaml('modalities.yaml', values)


def all_finite(data):
    assert isfinite(data).all()


def get_validator(name):
    assert name in locals()
    return locals()[name]
