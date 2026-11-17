"""PSO objective: F(x) = alpha * Acc_train + beta * Acc_val  (paper Eq. 4), alpha=0.7, beta=0.3.

Computed with k-fold stratified CV INSIDE the training part. The test set is never used here.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold

from .classifiers import make_classifier


class FitnessFunction:
    def __init__(self, X, y, clf_name, seed, clf_params=None, alpha=0.7, beta=0.3, k=4):
        self.X, self.y, self.clf_name, self.seed = X, y, clf_name, seed
        self.clf_params, self.alpha, self.beta, self.k = clf_params, alpha, beta, k
        self.folds = list(StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y))
        self.n_evals = 0

    def __call__(self, mask):
        mask = np.asarray(mask, bool)
        self.n_evals += 1
        if mask.sum() == 0:  # empty subset is infeasible -> worst fitness
            return 0.0
        Xs = self.X[:, mask]
        tr_acc, va_acc = [], []
        for tr, va in self.folds:
            clf = make_classifier(self.clf_name, self.seed, self.clf_params).fit(Xs[tr], self.y[tr])
            tr_acc.append(clf.score(Xs[tr], self.y[tr]))
            va_acc.append(clf.score(Xs[va], self.y[va]))
        return self.alpha * float(np.mean(tr_acc)) + self.beta * float(np.mean(va_acc))
