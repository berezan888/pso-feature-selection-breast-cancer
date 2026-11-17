"""The ten classifiers of the paper (sklearn defaults, Sect. 4.4) with optional overrides."""
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.exceptions import ConvergenceWarning
from sklearn.tree import DecisionTreeClassifier

# MLP with a small iteration budget does not always converge; that is expected, not an error.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

ALL_CLASSIFIERS = ["ADB", "DT", "KNN", "LDA", "LGR", "LR", "NB", "NN", "RF", "SVM"]


class LinearRegressionClassifier(ClassifierMixin, BaseEstimator):
    """'LR' in the paper: linear regression on 0/1 labels, class = (prediction > 0.5)."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.model_ = LinearRegression().fit(X, y)
        return self

    def decision_function(self, X):
        return self.model_.predict(X)

    def predict(self, X):
        return (self.decision_function(X) > 0.5).astype(int)


def make_classifier(name, seed, params=None):
    p = dict(params or {})
    factory = {
        "ADB": lambda: AdaBoostClassifier(random_state=seed, **p),
        "DT": lambda: DecisionTreeClassifier(random_state=seed, **p),
        "KNN": lambda: KNeighborsClassifier(**p),
        "LDA": lambda: LinearDiscriminantAnalysis(**p),
        "LGR": lambda: LogisticRegression(max_iter=p.pop("max_iter", 1000), **p),
        "LR": lambda: LinearRegressionClassifier(),
        "NB": lambda: GaussianNB(**p),
        "NN": lambda: MLPClassifier(random_state=seed, **p),
        "RF": lambda: RandomForestClassifier(random_state=seed, **p),
        "SVM": lambda: SVC(random_state=seed, **p),
    }
    if name not in factory:
        raise KeyError(f"unknown classifier {name}; choose from {ALL_CLASSIFIERS}")
    return factory[name]()


def score_function(clf):
    """Continuous score for ROC-AUC."""
    return clf.predict_proba if hasattr(clf, "predict_proba") else clf.decision_function
