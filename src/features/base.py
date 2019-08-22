from collections import defaultdict

import numpy as np
import pandas as pd

from tqdm import tqdm

from os.path import join, curdir

from .. import BaseGraph, FeatureMeta

from ..datasets import Dataset
from ..utils import feature_decorator, load_yaml, sliding_window_rect


def get_transformer(df):
    lookup = {}
    for col in df.columns:
        for val in df[col].unique():
            if val not in lookup:
                lookup[val] = len(lookup)
    return lookup, {vv: kk for kk, vv in lookup.items()}


def transform_df(df, lookup):
    df_copy = df.copy()
    for col in df_copy.columns:
        df_copy[col] = df_copy[col].apply(lambda dd: lookup[dd])
    return df_copy


class FeatureBase(BaseGraph):
    def __init__(self, name, parent, *args, **kwargs):
        super(FeatureBase, self).__init__(name=name)
        
        self.meta = FeatureMeta(name)
        self.parent = parent
        
        assert hasattr(parent, 'fs')
    
    @property
    def identifier(self):
        return join(
            self.parent.identifier,
            self.meta.name,
            self.meta.window.dir_name(),
        )
    
    def compose_index(self, *args, **kwargs):
        return self.node(
            node_name=self.build_path('index'),
            func=PartitionWindowExtractMetaData(
                win_len=self.meta.window.win_len,
                win_inc=self.meta.window.win_inc,
            ),
            sources=dict(
                index=self.parent.index,
                data=self.parent.index
            ),
            kwargs=dict(
                key='index',
                win_len=self.meta.window
            )
        )
    
    def compose_fold(self, *args, **kwargs):
        return self.node(
            node_name=self.build_path('fold'),
            func=PartitionWindowExtractMetaData(
                win_len=self.meta.window.win_len,
                win_inc=self.meta.window.win_inc,
            ),
            sources=dict(
                index=self.parent.index,
                data=self.parent.fold
            ),
            kwargs=dict(
                key='fold',
                win_len=self.meta.window
            )
        )
    
    def compose_label(self, *args, **kwargs):
        return self.node(
            node_name=self.build_path('label'),
            func=PartitionWindowExtractMetaData(
                win_len=self.meta.window.win_len,
                win_inc=self.meta.window.win_inc,
            ),
            sources=dict(
                index=self.parent.index,
                data=self.parent.label
            ),
            kwargs=dict(
                key='label',
                win_len=self.meta.window
            )
        )
    
    def compose_outputs(self):
        outputs = defaultdict(dict)
        
        def _compose_outputs(key):
            assert key not in outputs
            print(key, outputs.keys())
            modality, location = key[0], key[1]
            node = self.node(
                node_name=self.build_path(*key),
                func=PartitionWindowExtract(
                    win_len=self.meta.window.win_len,
                    win_inc=self.meta.window.win_inc,
                    feat_func=self.build_data,
                ),
                sources=dict(
                    index=self.parent.index,
                    data=self.parent.outputs[modality, location]
                ),
                kwargs=dict(
                    key=key,
                    fs=self.parent.dataset_meta.fs,
                )
            )
            return node
        
        if self.output_list is None:
            for kk, data in self.parent.iter_outputs():
                outputs[kk] = _compose_outputs(kk)
        else:
            for kk in self.output_list:
                outputs[kk] = _compose_outputs(kk)
        return outputs
