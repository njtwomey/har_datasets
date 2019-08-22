from os.path import join, exists
from os import symlink

from .. import BaseGraph, TransformerMeta

from ..utils import Partition


def make_symlink(in_path, out_path, ext):
    in_path = f'{in_path}.{ext}'
    out_path = f'{out_path}.{ext}'
    assert exists(in_path)
    if not exists(out_path):
        symlink(in_path, out_path)


class TransformerBase(BaseGraph):
    def __init__(self, name, parent, *args, **kwargs):
        super(TransformerBase, self).__init__(name=name)
        
        self.meta = TransformerMeta(name)
        self.parent = parent
        
        assert hasattr(parent.meta, 'fs')
    
    @property
    def identifier(self):
        return join(
            self.parent.identifier,
            self.name,
        )
    
    def make_node(self, in_key, out_key, func, backend=None):
        if not self.meta['resamples']:
            if in_key == out_key and isinstance(in_key, str) and in_key in {'index', 'label', 'fold'}:
                key = out_key
                return self.node(
                    node_name=self.build_path(key),
                    func=make_symlink,
                    kwargs=dict(
                        in_path=self.parent.build_path(key),
                        out_path=self.build_path(key),
                        ext=self.backends[self.default_backend].ext
                    ),
                )
        
        return self.node(
            node_name=self.build_path(*out_key),
            func=Partition(
                func=func,
            ),
            sources=dict(
                index=self.parent.index,
                data=self.parent.outputs[in_key]
            ),
            kwargs=dict(
                key=out_key,
                fs=self.parent.meta.fs,
                **(self.extra_args or dict())
            ),
        )
    
    def compose_index(self, *args, **kwargs):
        return self.make_node('index', 'index', None, 'none')
    
    def compose_fold(self, *args, **kwargs):
        return self.make_node('fold', 'fold', None, 'none')
    
    def compose_label(self, *args, **kwargs):
        return self.make_node('label', 'label', None, 'none')
    
    def compose_outputs(self):
        outputs = dict()
        for in_key, out_key, func in self.output_list:
            outputs[out_key] = self.make_node(in_key, out_key, func)
        return outputs
