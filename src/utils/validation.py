from .loaders import load_yaml

__all__ = [
    'check_yaml', 'check_activities', 'check_locations', 'check_modalities'
]


def check_yaml(name, values):
    value_set = load_yaml(name)
    for value in values:
        if value not in value_set:
            raise ValueError(f'{value} is not yet in `{name}.yaml`')
    if isinstance(values, dict):
        return {vv: kk for kk, vv in values.items()}


def check_activities(values):
    return check_yaml('activities', values)


def check_locations(values):
    return check_yaml('locations', values)


def check_modalities(values):
    return check_yaml('modalities', values)
