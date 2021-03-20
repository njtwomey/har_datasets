from src.features.statistical_features_impl import f_feat
from src.features.statistical_features_impl import t_feat

__all__ = ["statistical_features"]


def statistical_features(parent):
    """
    There are two feature categories defined here:
      1. Time domain
      2. Frequency domain

    And these get mapped from transformed data from two sources:
      1. Acceleration
      2. Gyroscope

    Assuming these two sources have gone through some body/gravity transformations
    (eg from src.transformations.body_grav_filt) there will actually be several
    more sources, eg:
      1. accel-body
      2. accel-body-jerk
      3. accel-body-jerk
      4. accel-grav
      5. gyro-body
      6. gyro-body-jerk
      7. gyro-body-jerk

    With more data sources this list will grows quickly.

    The feature types (time and frequency domain) are mapped to the transformed
    sources in a particular way. For example, the frequency domain features are
    *not* calculated on the gravity data sources. The loop below iterates through
    all of the outputs of the previous node in the graph, and the logic within
    the loop manages the correct mapping of functions to sources.

    Consult with the dataset table (tables/datasets.md) and see anguita2013 for
    details.
    """

    root = parent / "statistical_features"

    fs = root.get_ancestral_metadata("fs")

    accel_key = "mod='accel'"
    gyro_key = "mod='gyro'"
    mag_key = "mod='mag'"

    for key, node in parent.outputs.items():
        key_td = f"{key}-feat='td'"
        key_fd = f"{key}-feat='fd'"

        t_kwargs = dict(data=node)
        f_kwargs = dict(data=node, fs=fs)

        if accel_key in key:
            root.instantiate_node(key=key_td, func=t_feat, kwargs=t_kwargs)
            if "grav" not in key:
                root.instantiate_node(key=key_fd, func=f_feat, kwargs=f_kwargs)
        if gyro_key in key or mag_key in key:
            root.instantiate_node(key=key_td, func=t_feat, kwargs=t_kwargs)
            root.instantiate_node(key=key_fd, func=f_feat, kwargs=f_kwargs)

    return root
