from pathlib import Path

from loguru import logger

from src.utils.loaders import get_env
from src.utils.loaders import load_yaml
from src.utils.loaders import metadata_path


__all__ = [
    "DatasetMeta",
    "BaseMeta",
    "HARMeta",
    "PlacementMeta",
    "ModalityMeta",
    "DatasetMeta",
]


class BaseMeta(object):
    def __init__(self, path, *args, **kwargs):
        self.path = Path(path)
        self.name = self.path.stem
        self.meta = dict()

        if path:
            try:
                meta = load_yaml(path)

                if meta is None:
                    logger.info(
                        f'The content metadata module "{self.name}" from {path} is empty. Assigning empty dict'
                    )
                    meta = dict()
                else:
                    assert isinstance(meta, dict)

                self.meta = meta

            except FileNotFoundError:
                # logger.warn(f'The metadata file for "{self.name}" was not found.')
                pass

    def __getitem__(self, item):
        if item not in self.meta:
            logger.exception(KeyError(f"{item} not found in {self.__class__.__name__}"))
        return self.meta[item]

    def __contains__(self, item):
        return item in self.meta

    def __repr__(self):
        return f"<{self.name} {self.meta.__repr__()}>"

    def keys(self):
        return self.meta.keys()

    def values(self):
        return self.meta.values()

    def items(self):
        return self.meta.items()

    def insert(self, key, value):
        assert key not in self.meta
        self.meta[key] = value


"""
Non-functional metadata
"""


class HARMeta(BaseMeta):
    def __init__(self, path, *args, **kwargs):
        super(HARMeta, self).__init__(path=metadata_path("tasks", "har.yaml"), *args, **kwargs)


class LocalisationMeta(BaseMeta):
    def __init__(self, path, *args, **kwargs):
        super(LocalisationMeta, self).__init__(
            path=metadata_path("tasks", "localisation.yaml"), *args, **kwargs
        )


class PlacementMeta(BaseMeta):
    def __init__(self, path, *args, **kwargs):
        super(PlacementMeta, self).__init__(name=metadata_path("placement.yaml"), *args, **kwargs)


class ModalityMeta(BaseMeta):
    def __init__(self, path, *args, **kwargs):
        super(ModalityMeta, self).__init__(name=metadata_path("modality.yaml"), *args, **kwargs)


class DatasetMeta(BaseMeta):
    def __init__(self, path, *args, **kwargs):
        if isinstance(path, str):
            path = Path("metadata", "datasets", f"{path}.yaml")

        assert path.exists()

        super(DatasetMeta, self).__init__(path=path, *args, **kwargs)

        if "fs" not in self.meta:
            logger.exception(KeyError(f'The file {path} does not contain the key "fs"'))

        self.inv_lookup = dict()

        for task_name in self.meta["tasks"].keys():
            task_label_file = metadata_path("tasks", f"{task_name}.yaml")
            task_labels = load_yaml(task_label_file)
            dataset_labels = self.meta["tasks"][task_name]["target_transform"]
            if not set(dataset_labels.keys()).issubset(task_labels.keys()):
                logger.exception(
                    ValueError(
                        f"The following labels from dataset {path} are not accounted for in {task_label_file}: "
                        f"{set(dataset_labels.keys()).difference(task_labels.keys())}"
                    )
                )
            self.inv_lookup[task_name] = {
                dataset_labels[kk]: kk for kk, vv in dataset_labels.items()
            }

    @property
    def fs(self):
        return float(self.meta["fs"])

    @property
    def zip_path(self):
        return get_env("ZIP_ROOT") / self.name
