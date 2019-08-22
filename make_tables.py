# -*- coding: utf-8 -*-
import os

from src import DatasetMeta, load_datasets, dot_env_stuff
from src.utils.loaders import load_yaml, build_path


def make_dataset_row(dataset):
    def make_links(links, desc='Link'):
        return ', '.join(
            '[{} {}]({})'.format(desc, ii, url) for ii, url in enumerate(links, start=1)
        )
    
    # modalities = sorted(set([mn for ln, lm in self.meta['locations'].items() for mn, mv in lm.items() if mv]))
    
    data = [
        dataset.meta['author'],
        dataset.meta['paper_name'],
        dataset.name,
        make_links(links=dataset.meta['description_urls'], desc='Link'),
        dataset.meta.get('missing', ''),
        make_links(links=dataset.meta['paper_urls'], desc='Link'),
        dataset.meta['year'],
        dataset.meta['fs'],
        ', '.join(dataset.meta['locations'].keys()),
        ', '.join(dataset.meta['modalities']),
        dataset.meta['num_subjects'],
        dataset.meta['num_activities'],
        ', '.join(dataset.meta['activities'].keys()),
    ]
    
    return (
        (
            f'| First Author | Paper Name | Dataset Name | Description | Missing data '
            f'| Download Links | Year | Sampling Rate | Device Locations | Device Modalities '
            f'| Num Subjects | Num Activities | Activities | '
        ),
        '| {} |'.format(' | '.join(['-----'] * len(data))),
        '| {} |'.format(' | '.join(map(str, data)))
    )


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
        head, space, line = make_dataset_row(dataset)
        lines.append(line)
    with open(build_path('tables', 'datasets.md'), 'w') as fil:
        fil.write('{}\n'.format(head))
        fil.write('{}\n'.format(space))
        for line in lines:
            fil.write('{}\n'.format(line))
    
    # Iterate over the other data tables
    dims = [
        'modalities', 'activities', 'locations', 'representations',
        'features', 'models', 'transformers',
    ]
    
    for dim in dims:
        with open(build_path('tables', f'{dim}.md'), 'w') as fil:
            data = load_yaml(f'{dim}.yaml')
            fil.write(f'| Index | {dim[0].upper()}{dim[1:].lower()} | \n')
            fil.write('| ----- | ----- | \n')
            if not data or len(data) == 0:
                continue
            assert isinstance(data, dict)
            for ki, kv in data.items():
                fil.write(f'| {kv} | {ki} | \n')


if __name__ == '__main__':
    dot_env_stuff(main)
