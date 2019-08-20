from collections import defaultdict

import numpy as np
import pandas as pd

from os.path import join, curdir

from .. import BaseGraph

from ..datasets import Dataset
from ..utils import feature_decorator, load_yaml


class Window(object):
    def __init__(self, win_len=None, win_inc=None, *args, **kwargs):
        self.win_len = win_len
        self.win_inc = win_inc
    
    def dir_name(self):
        if self.win_len is None:
            return curdir
        return join(
            f'win_len={self.win_len}s',
            f'win_inc={self.win_inc}s',
        )


class FeatureMeta(object):
    def __init__(self, feature, *args, **kwargs):
        assert isinstance(feature, str)
        features = load_yaml('features.yaml')
        assert feature in features
        self.name = feature
        self.meta = features[feature]
    
    @property
    def windowed(self):
        return self.meta.get('windowed', False)
    
    @property
    def window(self):
        return Window(
            **self.meta['win_spec']
        )
    
    @property
    def modalities(self):
        return self.meta['modalities']


class PartitionWindowExtract(object):
    def __init__(self, win_len, win_inc, fs, feat_func):
        self.win_len = win_len
        self.win_inc = win_inc
        
        self.fs = fs
        
        self.feat_func = feat_func
    
    def __call__(self, index, data, *args, **kwargs):
        vals = index.trial.unique()
        
        values = []
        
        for val in vals:
            inds = index.trial == val
            
            feat = self.feat_func(
                index=index.iloc[inds],
                data=data.iloc[inds],
                *args, **kwargs
            )
            
            values.append(np.asarray(feat).ravel())
        
        return pd.DataFrame(values)


def index_iter(index, data):
    assert index.shape[0] == data.shape[0]
    vals = index.trial.unique()
    for val in vals:
        inds = index.index[index.trial == val]
        index_slice = index.iloc[inds]
        data_slice = index.iloc[inds]
        yield index_slice, data_slice


def build_slices(index, data, window):
    print(index.head())
    print(data.head())
    print('\n\n')


class FeatureBase(BaseGraph):
    def __init__(self, feature, dataset, outputs=None, *args, **kwargs):
        super(FeatureBase, self).__init__(feature)
        self.feature = FeatureMeta(feature)
        self.dataset = Dataset(dataset)
        
        self.index = None
        self.label = None
        self.fold = None
        
        self.output_list = outputs
        self.outputs = None
        
        self.compose()
    
    @property
    def identifier(self):
        return join(
            self.dataset.identifier,
            self.feature.name,
            self.feature.window.dir_name(),
        )
    
    def compose_index(self, *args, **kwargs):
        return self.node(
            node_name=self.build_path('index'),
            func=build_slices,
            sources=dict(
                index=self.dataset.index,
                data=self.dataset.index
            ),
            kwargs=dict(
                win_len=self.feature.window
            )
        )
    
    def compose_fold(self, *args, **kwargs):
        return self.node(
            node_name=self.build_path('fold'),
            func=build_slices,
            sources=dict(
                index=self.dataset.index,
                data=self.dataset.fold
            ),
            kwargs=dict(
                win_len=self.feature.window
            )
        )
    
    def compose_label(self, *args, **kwargs):
        return self.node(
            node_name=self.build_path('label'),
            func=build_slices,
            sources=dict(
                index=self.dataset.index,
                data=self.dataset.label
            ),
            kwargs=dict(
                win_len=self.feature.window
            )
        )
    
    def compose_outputs(self):
        outputs = defaultdict(dict)
        
        def _compose_outputs(key):
            assert key not in outputs
            
            modality, location = key[0], key[1]
            
            return self.node(
                node_name=self.build_path(*key),
                func=PartitionWindowExtract(
                    win_len=self.feature.window.win_len,
                    win_inc=self.feature.window.win_inc,
                    fs=self.dataset.dataset_meta.fs,
                    feat_func=self.build_data,
                ),
                sources=dict(
                    index=self.dataset.index,
                    data=self.dataset.outputs[modality][location]
                ),
                kwargs=dict(
                    modality=modality,
                    location=location,
                    fs=self.dataset.dataset_meta.fs,
                )
            )
        
        if self.output_list is None:
            for kk, data in self.dataset.iter_outputs():
                outputs[kk] = _compose_outputs(kk)
        else:
            for kk in self.output_list:
                outputs[kk] = _compose_outputs(kk)
        return outputs
    
    def build_data(self, modality, location, *args, **kwargs):
        raise NotImplementedError
