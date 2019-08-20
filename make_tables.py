# -*- coding: utf-8 -*-
import os

from src import DatasetMeta, load_datasets, dot_env_stuff
from src.utils.loaders import load_yaml, build_path


def main():
    # Ensure the paths exist
    root = build_path('tables')
    if not os.path.exists(root):
        os.makedirs(root)
    
    # Current list of datasets
    lines = []
    datasets = load_datasets()
    for dataset in datasets:
        dataset = DatasetMeta(dataset)
        head, space, line = dataset.make_row()
        lines.append(line)
    with open(build_path('tables', 'datasets.md'), 'w') as fil:
        fil.write('{}\n'.format(head))
        fil.write('{}\n'.format(space))
        for line in lines:
            fil.write('{}\n'.format(line))
    
    # Iterate over the other data tables
    for dim in ('modalities', 'activities', 'locations', 'representations', 'features', 'models'):
        with open(build_path('tables', f'{dim}.md'), 'w') as fil:
            data = load_yaml(dim)
            fil.write(f'| Index | {dim[0].upper()}{dim[1:].lower()} | \n')
            fil.write('| ----- | ----- | \n')
            if not data or len(data) == 0:
                continue
            assert isinstance(data, dict)
            for ki, kv in data.items():
                fil.write(f'| {kv} | {ki} | \n')


if __name__ == '__main__':
    dot_env_stuff(main)
