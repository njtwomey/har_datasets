import numpy as np

from src.functional.common import sorted_node_values


__all__ = [
    "ecdf",
]


def ecdf(parent, n_components):
    root = parent / f"feat='ecdf'-k={n_components}"

    for key, node in parent.outputs.items():
        root.instantiate_node(
            key=f"{key}-ecdf", func=calc_ecdf, kwargs=dict(n_components=n_components, data=node),
        )

    return root.instantiate_node(
        key="features", func=np.concatenate, args=[sorted_node_values(root.outputs)], kwargs=dict(axis=1)
    )


def calc_ecdf(data, n_components):
    return np.asarray([ecdf_rep(datum, n_components) for datum in data])


def ecdf_rep(data, components):
    """
    Taken from: https://github.com/nhammerla/ecdfRepresentation/blob/master/python/ecdfRep.py

    Parameters
    ----------
    data
    components

    Returns
    -------

    """

    m = data.mean(0)
    data = np.sort(data, axis=0)
    data = data[np.int32(np.around(np.linspace(0, data.shape[0] - 1, num=components))), :]
    data = data.flatten()
    return np.hstack((data, m))
