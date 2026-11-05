"""Simple baselines with the batch protocol (fit on the startup phase, predict; -1 = unknown).

OCND: one-class SVM on all known classes, trained once.
CND: MLP, unknown below max class probability `threshold`, self-trained on samples recognized as known.
"""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.svm import OneClassSVM


class OCND:

    def __init__(self, nu=0.5):
        self.nu = nu
        self.model = None
        self.known_label = None
        self.undefined_criteria = ("crit1", "crit3")

    def fit(self, X, y):
        self.model = OneClassSVM(nu=self.nu).fit(X)
        self.known_label = int(np.unique(y)[0])
        return self

    def predict(self, X):
        inlier = self.model.predict(np.asarray(X)) == 1
        return np.where(inlier, self.known_label, -1)


class CND:

    def __init__(self, threshold=0.9, random_state=None, **mlp_kwargs):
        self.threshold = threshold
        self.random_state = random_state
        self.mlp_kwargs = mlp_kwargs
        self.mlp = None
        self.single_class = None
        self.undefined_criteria = ("crit3",)

    def fit(self, X, y):
        classes = np.unique(y)
        if len(classes) < 2:
            self.single_class = int(classes[0])
            self.undefined_criteria = ("crit1", "crit2", "crit3")
            return self
        self.mlp = MLPClassifier(random_state=self.random_state, **self.mlp_kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.mlp.fit(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        if self.mlp is None:
            return np.full(len(X), self.single_class)
        proba = self.mlp.predict_proba(X)
        labels = self.mlp.classes_[np.argmax(proba, axis=1)]
        known = proba.max(axis=1) >= self.threshold
        if known.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                self.mlp.partial_fit(X[known], labels[known])
            # keep loss_curve_ from growing with the stream
            del self.mlp.loss_curve_[:-1]
        return np.where(known, labels, -1).astype(int)
