import json
import random

import numpy as np


__all__ = ["randomised_order", "NumpyEncoder"]


def randomised_order(iterable):
    iterable = list(iterable)
    random.shuffle(iterable)
    yield from iterable


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)
