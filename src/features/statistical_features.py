from src.features.statistical_features_impl import f_feat
from src.features.statistical_features_impl import t_feat

__all__ = ["statistical_features"]


def statistical_features(parent):
    root = parent / "statistical_features"

    kwargs = dict(fs=root.get_ancestral_metadata("fs"))

    # There are two feature categories defined here:
    #   1. Time domain
    #   2. Frequency domain
    #
    # And these get mapped from transformed data from two sources:
    #   1. Acceleration
    #   2. Gyroscope
    #
    # Assuming these two sources have gone through some body/gravity transformations
    # (eg from src.transformations.body_grav_filt) there will actually be several
    # more sources, eg:
    #   1. accel-body
    #   2. accel-body-jerk
    #   3. accel-body-jerk
    #   4. accel-grav
    #   5. gyro-body
    #   6. gyro-body-jerk
    #   7. gyro-body-jerk
    #
    # With more data sources this list will grows quickly.
    #
    # The feature types (time and frequency domain) are mapped to the transformed
    # sources in a particular way. For example, the frequency domain features are
    # *not* calculated on the gravity data sources. The loop below iterates through
    # all of the outputs of the previous node in the graph, and the logic within
    # the loop manages the correct mapping of functions to sources.
    #
    # Consult with the dataset table (tables/datasets.md) and see anguita2013 for
    # details.

    index = parent.index["index"]
    for key, node in parent.outputs.items():
        key_td = key + ("td",)
        key_fd = key + ("fd",)

        loop_kwargs = dict(index=index, data=node, **kwargs)

        if "accel" in key:
            root.outputs.add_output(key=key_td, func=t_feat, kwargs=loop_kwargs)
            if "grav" not in key:
                root.outputs.add_output(key=key_fd, func=f_feat, kwargs=loop_kwargs)
        if "gyro" in key:
            root.outputs.add_output(key=key_td, func=f_feat, kwargs=loop_kwargs)
            root.outputs.add_output(key=key_fd, func=f_feat, kwargs=loop_kwargs)

    return root
