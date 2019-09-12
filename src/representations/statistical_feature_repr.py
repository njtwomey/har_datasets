from src.datasets import load_dataset
from src.transformers import load_transformer
from src.features import load_feature


def statistical_feature_repr(name, *args, **kwargs):
    dataset = load_dataset(name)
    filtered = load_transformer('body_grav', parent=dataset)
    windowed = load_transformer('window_256_1', parent=filtered)
    features = load_feature('statistical_features', parent=windowed)
    return features
