from .base import EvaluationBase

__all__ = [
    'classification_metrics',
]


class classification_metrics(EvaluationBase):
    def __init__(self, parent):
        super(classification_metrics, self).__init__(
            name=self.__class__.__name__,
            parent=parent,
        )
