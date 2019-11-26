from src.base import BaseGraph

__all__ = [
    "EvaluationBase"
]


class EvaluationBase(BaseGraph):
    def __init__(self, name, parent, *args, **kwargs):
        super(EvaluationBase, self).__init__(
            name=name,
            parent=parent,
        )
