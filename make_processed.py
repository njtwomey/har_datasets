from src import load_datasets, dot_env_stuff, module_importer


def main():
    for name, meta in load_datasets().items():
        print('Processing {}'.format(name))
        proc = module_importer(name)
        proc.evaluate_all()


if __name__ == '__main__':
    dot_env_stuff(main)
