from .base import TransformerBase
from .. import Partition


def resample_data(key, index, data, fs_old, fs_new):
    print(key)
    print()


def resample_metadata(key, index, data, fs_old, fs_new):
    print(key)
    print()


class Resampler(TransformerBase):
    def __init__(self, name, parent, fs_new):
        super(Resampler, self).__init__(name=name, parent=parent)
        
        kwargs = dict(
            fs_old=parent.get_ancestral_metadata('fs'),
            fs_new=fs_new
        )
        
        for key, node in parent.index.items():
            self.index.add_output(
                key=key,
                func=Partition(resample_metadata),
                sources=dict(
                    index=parent.index['index'],
                    data=node,
                ),
                **kwargs
            )
        
        for key, node in parent.outputs.items():
            self.outputs.add_output(
                key=key,
                func=Partition(resample_data),
                sources=dict(
                    index=parent.index['index'],
                    data=node,
                ),
                **kwargs
            )


class resample_33(Resampler):
    def __init__(self, parent):
        super(resample_33, self).__init__(
            name=self.__class__.__name__,
            parent=parent,
            fs_new=33
        )
