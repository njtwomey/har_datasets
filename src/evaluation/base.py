from mldb.backends import JsonBackend

from .. import BaseGraph, EvaluationMeta

__all__ = [
    'EvaluationBase',
]


class EvaluationBase(BaseGraph):
    def __init__(self, name, parent, *args, **kwargs):
        super(EvaluationBase, self).__init__(
            name=name,
            parent=parent,
            meta=EvaluationMeta(name),
        )
        
        self.add_backend('json', JsonBackend(path=self.fs_root), default=True)
