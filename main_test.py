from src import dot_env_decorator

from src.utils import take_all, accel_filt, gyro_filt, randomised_order

from src import dataset_importer
from src import features
from src import transformers
from src import selectors
from src import models


@dot_env_decorator
def main(
    dataset_name, source_filter, feat_name, clf_name, fs_new, win_len, win_inc, split_type,
):
    dataset = dataset_importer(dataset_name)
    resampled = transformers.resample(parent=dataset, fs_new=fs_new)
    filtered = transformers.body_grav_filter(parent=resampled)
    windowed = transformers.window(parent=filtered, win_len=win_len, win_inc=win_inc)
    if feat_name == "ecdf":
        feats = features.ecdf(parent=windowed, source_filter=source_filter, n_components=21)
    elif feat_name == "statistical":
        feats = features.statistical_features(parent=windowed, source_filter=source_filter,)
    else:
        raise ValueError
    task = selectors.select_task(parent=feats, task_name="har")
    split = selectors.select_split(parent=task, split_type=split_type)
    if clf_name == "sgd_classifier":
        clf = models.sgd_classifier(parent=split, data=feats)
    elif clf_name == "logistic_regression_cv":
        clf = models.logistic_regression_cv(parent=split, data=feats)
    else:
        raise ValueError
    return clf.get_results()


if __name__ == "__main__":
    from sklearn.model_selection import ParameterGrid
    import pandas as pd

    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)

    configurations = dict(
        dataset_name=[
            "anguita2013",
            # 'pamap2',
            # 'uschad',
        ],
        feat_name=["ecdf", "statistical"],
        clf_name=["sgd_classifier", "logistic_regression_cv"],
        fs_new=[33],
        win_len=[2.56, 5.12],
        win_inc=[1.0],
        source_filter=[accel_filt, gyro_filt, take_all],
        split_type=["predefined"],
    )

    results = []
    for config in randomised_order(ParameterGrid(configurations)):
        result = main(**config)
        results.append(
            dict(
                **pd.DataFrame(
                    [
                        {k: v for k, v in result[fold]["test"].items() if k not in {"xval"}}
                        for fold in result.keys()
                    ]
                )
                .mean(0)
                .to_dict(),
                **config,
            )
        )

    df = pd.DataFrame(results)

    print(
        df.groupby("dataset_name").apply(lambda gg: gg.sort_values(by="accuracy", ascending=False))
    )
