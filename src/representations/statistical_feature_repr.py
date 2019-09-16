from src.datasets import load_dataset
from src.transformers import body_grav_filter, window_256_1, resample_33
from src.features import statistical_features, ecdf_11, ecdf_21
from src.visualisations import umap_embedding


def statistical_feature_repr(name):
    dataset = load_dataset(name)
    # data_33hz = resample_33(parent=dataset)
    filtered = body_grav_filter(parent=dataset)
    windowed = window_256_1(parent=filtered)
    stat = ecdf_21(parent=windowed)
    viz = umap_embedding(parent=stat)
    return viz
