from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from src.models.sklearn.base import sklearn_model

__all__ = [
    'scale_log_reg', 'scale_sgd_clf', 'random_forest'
]


class scale_log_reg(sklearn_model):
    def __init__(self, parent):
        super(scale_log_reg, self).__init__(
            name=self.__class__.__name__,
            parent=parent,
            model=Pipeline((
                ('scale', StandardScaler()),
                ('clf', LogisticRegression())
            ))
        )


class scale_sgd_clf(sklearn_model):
    def __init__(self, parent):
        super(scale_sgd_clf, self).__init__(
            name=self.__class__.__name__,
            parent=parent,
            model=Pipeline((
                ('scale', StandardScaler()),
                ('clf', SGDClassifier())
            ))
        )


class random_forest(sklearn_model):
    def __init__(self, parent):
        super(random_forest, self).__init__(
            name=self.__class__.__name__,
            parent=parent,
            model=Pipeline((
                ('clf', RandomForestClassifier(n_estimators=10))
            ))
        )


class knn(sklearn_model):
    def __init__(self, parent):
        super(knn, self).__init__(
            name=self.__class__.__name__,
            parent=parent,
            model=Pipeline((
                ('scale', StandardScaler()),
                ('clf', KNeighborsClassifier()),
            )),
            save_model=False,
        )


class svm(sklearn_model):
    def __init__(self, parent):
        super(svm, self).__init__(
            name=self.__class__.__name__,
            parent=parent,
            model=Pipeline((
                ('scale', MinMaxScaler(feature_range=(-1, 1))),
                ('clf', SVC()),
            )),
            save_model=False,
        )
