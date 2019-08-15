from src import load_datasets, download_dataset, dot_env_stuff


def main():
    for name, meta in load_datasets().items():
        print('Downloading {}'.format(name))
        download_dataset(meta)


if __name__ == '__main__':
    dot_env_stuff(main)
