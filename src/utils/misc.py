from dotenv import find_dotenv, load_dotenv
import random
import json

from numpy import ndarray


__all__ = [
    'dot_env_stuff', 'randomised_order', 'NumpyEncoder'
]


def dot_env_stuff(func):
    """
    
    Args:
        func:

    Returns:

    """
    # log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # logging.basicConfig(level=logging.INFO, format=log_fmt)
    # project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    load_dotenv(find_dotenv())
    
    func()


def randomised_order(iterable):
    """
    
    Args:
        iterable:

    Returns:

    """
    iterable = list(iterable)
    random.shuffle(iterable)
    yield from iterable


class NumpyEncoder(json.JSONEncoder):
    """
    This simple JSONEncoder class allows numpy ndarrays to be serialised to file.
    This is useful for dumping the cross-validation data resulting from a grid search
    to a JSON record.
    """
    
    def default(self, obj):
        """
        This function allows numpy ndarrays to be serialised. Call with:

        ```python
        import numpy as np
        import json

        d = dict(one=np.linspace(0, 1, 7))
        # print(json.dumps(d))  # raises error
        print(json.dumps(d, cls=NumpyEncoder))  # success
        ```

        :param obj: The object to be serialised
        :return: Encoder object
        """
        if isinstance(obj, ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
