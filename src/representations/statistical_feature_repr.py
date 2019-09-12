from src.datasets import load_dataset
from src.transformers import body_grav, window_256_1
from src.features import statistical_features


def statistical_feature_repr(name):
    dataset = load_dataset(name)
    filtered = body_grav(parent=dataset)
    windowed = window_256_1(parent=filtered)
    stat = statistical_features(parent=windowed)
    return stat
