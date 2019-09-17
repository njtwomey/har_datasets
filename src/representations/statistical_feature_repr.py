from src.datasets import load_dataset
from src.transformers import body_grav_filter, window_256_1, resample_33
from src.features import statistical_features, ecdf_11, ecdf_21
from src.visualisations import umap_embedding
from src.models import scale_log_reg


def statistical_feature_repr(name):
    dataset = load_dataset(name)
    # data_33hz = resample_33(parent=dataset)
    filtered = body_grav_filter(parent=dataset)
    windowed = window_256_1(parent=filtered)
    features = statistical_features(parent=windowed)
    # umap = umap_embedding(parent=features)
    clf = scale_log_reg(parent=features)
    return clf


# def statistical_feature_repr(name):
#     windowed = statistical_features(parent=windowed)
#     return stat
#
#
# def statistical_feature_repr(name):
#     windowed = window_256_1(parent=filtered)
#     feats = statistical_features(parent=windowed)
#     return feats
#
#
# def statistical_feature_repr(name):
#     windowed = window_256_1(parent=filtered)
#     feats = statistical_features(parent=windowed)
#     return feats
