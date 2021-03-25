from typing import Dict
from typing import List
from typing import Optional
from typing import Sized
from typing import Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder


class PrefittedVotingClassifier(BaseEstimator):
    def __init__(
        self,
        estimators: List[Union[BaseEstimator]],
        voting: str = "soft",
        weights: Optional[Sized] = None,
        verbose: bool = False,
        strict: bool = True,
    ):
        assert weights is None or len(weights) == len(estimators)

        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.verbose = verbose
        self.strict = strict
        self.le_ = None
        self.classes_ = None

    def transform(self, X):
        weights = self.weights
        if weights is None:
            weights = np.ones(len(self.estimators)) / len(self.estimators)
        return [est.predict_proba(X) * ww for ww, (_, est) in zip(weights, self.estimators)]

    def predict_proba(self, X):
        return sum(self.transform(X))

    def predict(self, X):
        probs = self.predict_proba(X)
        inds = np.argmax(probs, axis=1)
        return self.classes_[inds]

    def fit(self, X, y, sample_weight=None):
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        for name, est in self.estimators:
            if self.strict:
                assert np.all(
                    est.classes_ == self.classes_
                ), f"Model classes ({self.classes_}) not aligned with {name}: {est.classes_=}"
        return self

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


class ZeroShotVotingClassifier(PrefittedVotingClassifier):
    def __init__(
        self,
        estimators: List[Union[BaseEstimator]],
        label_alignment: Dict[str, str],
        voting: str = "soft",
        weights: Optional[Sized] = None,
        verbose: bool = False,
    ):
        super().__init__(estimators=estimators, voting=voting, weights=weights, verbose=verbose, strict=False)
        self.label_alignment = label_alignment

    def predict_proba(self, X):
        out = np.zeros((X.shape[0], self.classes_.shape[0]))
        self_lookup = dict(zip(self.classes_, range(len(self.classes_))))
        for (_, estimator), transformed in zip(self.estimators, self.transform(X)):
            for fi, (name, col) in enumerate(zip(estimator.classes_, transformed.T)):
                out[:, self_lookup[self.label_alignment[name]]] += col
        return out

    def predict(self, X):
        probs = self.predict_proba(X)
        inds = np.argmax(probs, axis=1)
        return self.classes_[inds]
