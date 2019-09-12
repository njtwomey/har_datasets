from src import load_datasets, dot_env_stuff, dataset_importer


def main():
    for name, meta in load_datasets().items():
        proc = dataset_importer(name)
        proc.evaluate_outputs()


if __name__ == '__main__':
    dot_env_stuff(main)
