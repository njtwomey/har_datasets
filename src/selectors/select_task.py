from pathlib import Path

from src.selectors.base import SelectorBase
from src.utils.loaders import get_yaml_file_list
from src.utils.loaders import metadata_path

__all__ = [
    "select_task",
]


def task_selector(key, data):
    return data


class select_task(SelectorBase):
    def __init__(self, parent, task_name):
        tasks = get_yaml_file_list("tasks", stem=True)

        assert task_name in tasks

        super(select_task, self).__init__(
            name=task_name, parent=parent, meta=metadata_path("tasks", f"{task_name}.yaml"),
        )

        self.index.add_output(
            key="target",
            func=task_selector,
            backend="pandas",
            kwargs=dict(data=parent.index[task_name]),
        )
