from src import load_datasets, download_dataset, dot_env_stuff


def main():
    for name in load_datasets().keys():
        print('Downloading {}'.format(name))
        download_dataset(name)


if __name__ == '__main__':
    dot_env_stuff(main)
