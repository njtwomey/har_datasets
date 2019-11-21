from mldb.backends import JoblibBackend
from src.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    'ScikitLearnBackend',
]


class ScikitLearnBackend(JoblibBackend):
    def __init__(self, path):
        super(ScikitLearnBackend, self).__init__(ext='sklearn', path=path)
