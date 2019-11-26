import zipfile
from src import load_datasets_metadata, dot_env_decorator
from os.path import basename, join, split, exists, splitext
from os import makedirs
from tqdm import tqdm

import requests

from src import DatasetMeta


def unzip_data(zip_path, in_name, out_name):
    if exists(join(zip_path, out_name)):
        return
    with zipfile.ZipFile(join(zip_path, in_name), 'r') as fil:
        fil.extractall(zip_path)


def download_and_save(url, path, force=False, chunk_size=2 ** 12):
    response = requests.get(url, stream=True)
    fname = join(path, split(url)[1])
    desc = f'Downloading {fname}...'
    if exists(fname):
        if not force:
            return
    chunks = tqdm(
        response.iter_content(chunk_size=chunk_size), desc=basename(desc)
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
        zip_name = basename(dataset.meta['download_urls'][0])
        unzip_path = join(dataset.zip_path, splitext(zip_name)[0])
        unzip_data(
            zip_path=dataset.zip_path,
            in_name=zip_name,
            out_name=unzip_path
        )


@dot_env_decorator
def main():
    for name in load_datasets_metadata().keys():
        print('Downloading {}'.format(name))
        download_dataset(name)


if __name__ == '__main__':
    main()
