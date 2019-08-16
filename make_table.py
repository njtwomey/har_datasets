# -*- coding: utf-8 -*-
import os
from collections import namedtuple

from src import DatasetMeta, load_datasets, dot_env_stuff
from src.utils.loaders import load_yaml

ColumnPair = namedtuple('ColumnPair', ('short', 'long'))


def main():
    datasets = load_datasets()
    
    lines = []
    for dataset in datasets:
        dataset = DatasetMeta(dataset)
        head, space, line = dataset.make_row()
        lines.append(line)
    
    root = os.path.join(
        os.environ['PROJECT_ROOT'], 'tables'
    )
    
    if not os.path.exists(root):
        os.makedirs(root)
    
    # Current list of datasets
    with open(os.path.join(root, 'datasets.md'), 'w') as fil:
        fil.write('{}\n'.format(head))
        fil.write('{}\n'.format(space))
        for line in lines:
            fil.write('{}\n'.format(line))
    
    for dim in ('modalities', 'activities', 'locations'):
        with open(os.path.join(root, f'{dim}.md'), 'w') as fil:
            data = load_yaml(dim)
            fil.write(f'| Index | {dim[0].upper()}{dim[1:].lower()} | \n')
            fil.write('| ----- | ----- | \n')
            assert isinstance(data, dict)
            for ki, kv in data.items():
                fil.write(f'| {kv} | {ki} | \n')


if __name__ == '__main__':
    dot_env_stuff(main)
