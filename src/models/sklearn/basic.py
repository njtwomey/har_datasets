from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

import numpy as np

from src.models.sklearn.base import sklearn_model_factory

__all__ = ["logistic_regression", "logistic_regression_cv", "sgd_classifier"]


class logistic_regression(sklearn_model_factory):
    def __init__(self, parent, *args, **kwargs):
        super(logistic_regression, self).__init__(
            name=self.__class__.__name__,
            parent=parent,
            model=Pipeline((("scale", StandardScaler()), ("clf", LogisticRegression()))),
            *args,
            **kwargs,
        )


class logistic_regression_cv(sklearn_model_factory):
    def __init__(self, parent, *args, **kwargs):
        super(logistic_regression_cv, self).__init__(
            name=self.__class__.__name__,
            parent=parent,
            model=Pipeline((("scale", StandardScaler()), ("clf", LogisticRegressionCV()))),
            xval=dict(clf__multi_class=["multinomial"],),
            *args,
            **kwargs,
        )


class sgd_classifier(sklearn_model_factory):
    def __init__(self, parent, data, *args, **kwargs):
        super(sgd_classifier, self).__init__(
            name=self.__class__.__name__,
            parent=parent,
            data=data,
            model=Pipeline((("scale", StandardScaler()), ("clf", SGDClassifier()))),
            xval=dict(
                clf__loss=["log"],
                clf__penalty=["l2"],
                clf__alpha=np.power(10.0, np.arange(-5, 5 + 1)),
            ),
        )


class random_forest(sklearn_model_factory):
    def __init__(self, parent, *args, **kwargs):
        super(random_forest, self).__init__(
            name=self.__class__.__name__,
            parent=parent,
            model=Pipeline((("clf", RandomForestClassifier(n_estimators=10)))),
            *args,
            **kwargs,
        )


class knn(sklearn_model_factory):
    def __init__(self, parent, *args, **kwargs):
        super(knn, self).__init__(
            name=self.__class__.__name__,
            parent=parent,
            model=Pipeline((("scale", StandardScaler()), ("clf", KNeighborsClassifier()),)),
            *args,
            **kwargs,
        )


class svm(sklearn_model_factory):
    def __init__(self, parent, *args, **kwargs):
        super(svm, self).__init__(
            name=self.__class__.__name__,
            parent=parent,
            model=Pipeline((("scale", MinMaxScaler(feature_range=(-1, 1))), ("clf", SVC()),)),
            *args,
            **kwargs,
        )
