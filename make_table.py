# -*- coding: utf-8 -*-
import os
from collections import namedtuple

from src import DatasetMeta, load_datasets, dot_env_stuff

ColumnPair = namedtuple('ColumnPair', ('short', 'long'))


def main():
    datasets = load_datasets()
    
    lines = []
    for dataset in datasets:
        dataset = DatasetMeta(dataset)
        head, space, line = dataset.make_row()
        lines.append(line)
    
    with open(os.path.join(os.environ['PROJECT_ROOT'], 'datasets.md'), 'w') as fil:
        fil.write('{}\n'.format(head))
        fil.write('{}\n'.format(space))
        for line in lines:
            fil.write('{}\n'.format(line))


if __name__ == '__main__':
    dot_env_stuff(main)
