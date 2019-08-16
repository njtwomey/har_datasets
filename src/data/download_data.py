from os.path import join, split, exists
from os import makedirs
from tqdm import tqdm

import requests

from src import DatasetMeta


def download_and_save(url, path, force=False, chunk_size=2 ** 12):
    response = requests.get(url, stream=True)
    fname = join(path, split(url)[1])
    desc = f'Downloading {fname}...'
    if exists(fname):
        if not force:
            return
    chunks = tqdm(
        response.iter_content(chunk_size=chunk_size), desc=desc
    )
    with open(fname, 'wb') as fil:
        for chunk in chunks:
            fil.write(chunk)


def download_dataset(meta):
    dataset = DatasetMeta(meta)
    if not exists(dataset.zip_path):
        makedirs(dataset.zip_path)
    for ii, url in enumerate(dataset.meta['download_urls']):
        print('\t{}/{} {}'.format(ii + 1, len(dataset.meta['download_urls']), url))
        download_and_save(url=url, path=dataset.zip_path)
