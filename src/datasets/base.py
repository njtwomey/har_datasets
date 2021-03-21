from os.path import basename
from os.path import join
from os.path import splitext

import pandas as pd

from src.base import ExecutionGraph
from src.base import get_ancestral_metadata
from src.meta import DatasetMeta

__all__ = [
    "Dataset",
]


def validate_split_names(split_name, split_df, split_cols):
    meta_set = set(split_cols)
    df_set = set(split_df.columns)
    if not df_set.issubset(meta_set):
        raise ValueError(
            f"Split dataframe columns ({df_set}) not subset of metadata ({meta_set}) for the split type {split_name}."
        )


class Dataset(ExecutionGraph):
    def __init__(self, name, *args, **kwargs):
        super(Dataset, self).__init__(name=f"datasets/{name}", meta=DatasetMeta(name))

        def load_meta(*args, **kwargs):
            return self.meta.meta

        load_meta.__name__ = name

        metadata = self.instantiate_node(key=f"{name}-metadata", backend="yaml", func=load_meta, kwargs=dict())
        metadata.evaluate()

        zip_name = kwargs.get("unzip_path", lambda x: x)(splitext(basename(self.meta.meta["download_urls"][0]))[0])
        self.unzip_path = join(self.meta.zip_path, splitext(zip_name)[0])

        index = self.instantiate_node(
            key="index", func=self.build_index, backend="pandas", kwargs=dict(path=self.unzip_path, metatdata=metadata),
        )

        # Build the indexes
        predefined = self.instantiate_node(
            key="predefined",
            func=self.build_predefined,
            backend="pandas",
            kwargs=dict(path=self.unzip_path, metatdata=metadata),
        )
        predefined.evaluate()

        split_defs = get_ancestral_metadata(self, "splits")

        self.instantiate_node(
            key="loso", func=self.build_loso, backend="pandas", kwargs=dict(index=index, columns=split_defs["loso"]),
        )

        self.instantiate_node(
            key="deployable",
            func=self.build_deployable,
            backend="pandas",
            kwargs=dict(index=index, columns=split_defs["deployable"]),
        )

        tasks = get_ancestral_metadata(self, "tasks")
        for task in tasks:
            self.instantiate_node(
                key=task,
                func=self.build_label,
                backend="pandas",
                kwargs=dict(path=self.unzip_path, task=task, inv_lookup=self.meta.inv_lookup[task], metatdata=metadata),
            )

        # Build list of outputs
        for placement_modality in self.meta["sources"]:
            loc = placement_modality["loc"]
            mod = placement_modality["mod"]

            self.instantiate_node(
                key=f"{loc=}-{mod=}",
                func=self.build_data,
                backend="numpy",
                kwargs=dict(loc=loc, mod=mod, metadata=metadata),
            )

    def build_label(self, *args, **kwargs):
        raise NotImplementedError

    def build_index(self, *args, **kwargs):
        raise NotImplementedError

    def build_data(self, loc, mod, *args, **kwargs):
        raise NotImplementedError

    def build_predefined(self, *args, **kwargs):
        raise NotImplementedError

    def build_deployable(self, index, columns):
        splits = pd.DataFrame({"fold_0": ["train"] * len(index)}, dtype="category")
        validate_split_names(split_name="deployable", split_df=splits, split_cols=columns)
        return splits

    def build_loso(self, index, columns):
        splits = pd.DataFrame(
            {
                f"fold_{ki}": index.trial.apply(lambda tt: ["train", "test"][tt == kk])
                for ki, kk in enumerate(index.trial.unique())
            }
        )
        validate_split_names(split_name="loso", split_df=splits, split_cols=columns)
        return splits
