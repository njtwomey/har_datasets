from pathlib import Path

import pandas as pd

__all__ = [
    "select_split",
]


def predefined_split(split):
    return split.astype("category")


def deployable_split(split):
    return pd.DataFrame({"fold_0": ["train"] * len(split)}, dtype="category")


def loso_split(split):
    return pd.DataFrame(
        {
            f"fold_{kk}": split.trial.apply(lambda tt: ["train", "test"][tt == kk])
            for kk in split.trial.unique()
        }
    )


def select_split(parent, split_type):
    root = parent.make_child(name=split_type, meta=Path("metadata", "split.yaml"))

    assert split_type in root.meta["supported"]

    func_dict = dict(predefined=predefined_split, deployable=deployable_split, loso=loso_split)

    split = parent.index["index"]
    if split_type == "predefined":
        split = parent.index["fold"]

    root.index.add_output(
        key="split", func=func_dict[split_type], backend="pandas", kwargs=dict(split=split)
    )

    return root
