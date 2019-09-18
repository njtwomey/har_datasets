from src.datasets import load_dataset
from src.transformers import body_grav_filter, window_256_1, resample_33
from src.features import statistical_features, ecdf_11, ecdf_21
from src.visualisations import umap_embedding
from src.models import scale_log_reg, random_forest, knn, svm

__all__ = [
    'statistical_feature_repr'
]


def window_dataset(dataset, resample=False):
    if resample:
        dataset = resample_33(parent=dataset)
    filtered = body_grav_filter(parent=dataset)
    windowed = window_256_1(parent=filtered)
    return windowed


def statistical_feature_repr(name):
    dataset = load_dataset(name)
    windowed = window_dataset(dataset)
    features = statistical_features(parent=windowed)
    return features
