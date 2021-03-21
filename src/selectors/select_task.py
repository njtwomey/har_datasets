# from src.base import get_ancestral_metadata
# from src.utils.loaders import metadata_path
#
# __all__ = [
#     "select_task",
# ]
#
#
# def select_task_labels(data):
#     return data
#
#
# def select_task(parent, task_name):
#     root = parent.make_child(name=task_name, meta=metadata_path("tasks", f"{task_name}.yaml"))
#
#     assert task_name in get_ancestral_metadata(root, "tasks").keys()
#
#     root.instantiate_node(
#         key="target", func=select_task_labels, backend="pandas", kwargs=dict(data=parent.index[task_name]),
#     )
#
#     return root
