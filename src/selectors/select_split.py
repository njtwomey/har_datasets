from pathlib import Path

import pandas as pd

__all__ = [
    "select_split",
]

from src import ExecutionGraph


def validate_split_names(split_name, split_df, split_cols):
    meta_set = set(split_cols)
    df_set = set(split_df.columns)
    if not df_set.issubset(meta_set):
        raise ValueError(
            f"Split dataframe columns ({df_set}) not subset of metadata ({meta_set}) for the split type {split_name}."
        )


def predefined_split(split, columns):
    splits = split.astype("category")
    validate_split_names(split_name="predefined", split_df=splits, split_cols=columns)
    return splits


def deployable_split(split, columns):
    splits = pd.DataFrame({"fold_0": ["train"] * len(split)}, dtype="category")
    validate_split_names(split_name="deployable", split_df=splits, split_cols=columns)
    return splits


def loso_split(split, columns):
    splits = pd.DataFrame(
        {f"fold_{kk}": split.trial.apply(lambda tt: ["train", "test"][tt == kk]) for kk in split.trial.unique()}
    )
    validate_split_names(split_name="loso", split_df=splits, split_cols=columns)
    return splits


def select_split(parent: ExecutionGraph, split_type):
    root = parent.make_child(name=split_type, meta=Path("metadata", "split.yaml"))

    assert split_type in root.meta["supported"]

    func_dict = dict(predefined=predefined_split, deployable=deployable_split, loso=loso_split)

    split = parent.index["index"]
    if split_type == "predefined":
        split = parent.index["fold"]

    split_defs = root.get_ancestral_metadata("splits")
    root.instantiate_node(
        key="split",
        func=func_dict[split_type],
        backend="pandas",
        kwargs=dict(split=split, columns=split_defs[split_type]),
    )

    return root
