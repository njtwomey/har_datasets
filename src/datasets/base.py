from os.path import basename
from os.path import join
from os.path import splitext

from src.base import BaseGraph
from src.meta import DatasetMeta

__all__ = [
    "Dataset",
]


class Dataset(BaseGraph):
    def __init__(self, name, *args, **kwargs):
        super(Dataset, self).__init__(name=name, parent=None, meta=DatasetMeta(name))

        zip_name = kwargs.get("unzip_path", lambda x: x)(
            splitext(basename(self.meta.meta["download_urls"][0]))[0]
        )
        self.unzip_path = join(self.meta.zip_path, splitext(zip_name)[0])

        # Build the indexes
        self.index.add_output(
            key="fold", func=self.build_fold, backend="pandas", kwargs=dict(path=self.unzip_path)
        )

        self.index.add_output(
            key="index", func=self.build_index, backend="pandas", kwargs=dict(path=self.unzip_path,)
        )

        tasks = self.get_ancestral_metadata("tasks")
        for task in tasks:
            self.index.add_output(
                key=task,
                func=self.build_label,
                backend="pandas",
                kwargs=dict(
                    path=self.unzip_path, task=task, inv_lookup=self.meta.inv_lookup[task],
                ),
            )

        # Build list of outputs
        for placement_modality in self.meta["sources"]:
            placement = placement_modality["placement"]
            modality = placement_modality["modality"]

            self.outputs.add_output(
                key=(modality, placement), func=self.build_data, backend="numpy",
            )

    def build_label(self, *args, **kwargs):
        raise NotImplementedError

    def build_index(self, *args, **kwargs):
        raise NotImplementedError

    def build_fold(self, *args, **kwargs):
        raise NotImplementedError

    def build_data(self, *args, **kwargs):
        raise NotImplementedError
