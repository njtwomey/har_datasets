from os.path import basename
from os.path import join
from os.path import splitext

from src.base import ExecutionGraph
from src.meta import DatasetMeta

__all__ = [
    "Dataset",
]

# BUILD_ROOT = ExecutionGraph.build_root()


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

        # Build the indexes
        self.instantiate_node(
            key="fold", func=self.build_fold, backend="pandas", kwargs=dict(path=self.unzip_path, metatdata=metadata),
        )

        self.instantiate_node(
            key="index", func=self.build_index, backend="pandas", kwargs=dict(path=self.unzip_path, metatdata=metadata),
        )

        tasks = self.get_ancestral_metadata("tasks")
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

    def build_fold(self, *args, **kwargs):
        raise NotImplementedError

    def build_data(self, loc, mod, *args, **kwargs):
        raise NotImplementedError
