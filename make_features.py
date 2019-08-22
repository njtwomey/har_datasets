from src import load_datasets, load_features, dot_env_stuff, dataset_importer, feature_importer


def main():
    asdfasdfasdf
    for name in load_datasets():
        for feat_name in load_features():
            print(name, feat_name)
            features = feature_importer(
                class_name=feat_name,
                dataset=name
            )
            features.compose()
            features.evaluate_all()


if __name__ == '__main__':
    dot_env_stuff(main)
