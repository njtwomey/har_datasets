from src.selectors.base import SelectorBase
from src.utils.loaders import get_yaml_file_list

__all__ = [
    'select_task',
]


def task_selector(key, data):
    return data


class select_task(SelectorBase):
    def __init__(self, parent, task_name):
        super(select_task, self).__init__(
            name=task_name, parent=parent
        )
        
        tasks = get_yaml_file_list('tasks', strip_ext=True)
        
        assert task_name in tasks
        
        self.index.add_output(
            key='target',
            func=task_selector,
            data=parent.index[task_name],
            backend='pandas'
        )
