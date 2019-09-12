from src import load_transformations, load_datasets, dot_env_stuff, dataset_importer, transformer_importer


def main():
    for dataset in load_datasets():
        dataset = dataset_importer(dataset)
        for transformer in load_transformations():
            transformer = transformer_importer(transformer, parent=dataset)
            transformer.evaluate_all()


if __name__ == '__main__':
    dot_env_stuff(main)
