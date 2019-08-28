from src.datasets import load_dataset
from src.transformers import load_transformer
from src.features import load_feature


def statistical_feature_repr(name, *args, **kwargs):
    dataset = load_dataset(name)
    dataset.compose_check()
    
    filtered = load_transformer('body_grav', parent=dataset)
    filtered.compose()
    
    features = load_feature('statistical_features', parent=filtered)
    features.compose()
    
    return features
