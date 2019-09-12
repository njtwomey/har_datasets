from src import dot_env_stuff, load_datasets, load_representation


def main():
    datasets = load_datasets()
    for name in datasets.keys():
        representation = load_representation(
            'statistical_feature_repr', name=name
        )
        representation.evaluate_outputs()


if __name__ == '__main__':
    dot_env_stuff(main)
