from src import dot_env_stuff, load_datasets, load_representations, load_representation


def main():
    datasets = load_datasets()
    representations = load_representations()
    for name in datasets.keys():
        for representation in representations:
            print(name, representation)
            load_representation(
                representation, name=name
            ).evaluate_outputs()
        break


if __name__ == '__main__':
    dot_env_stuff(main)
