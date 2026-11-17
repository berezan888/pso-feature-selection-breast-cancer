"""Data loading, stratified splitting and (leak-free) preprocessing.

Paper pipeline (Fig. 3): missing values -> outliers -> min-max normalisation -> 80/20 split.
The paper applies the preprocessing BEFORE splitting (statistics from all data). We support both:
  fit_on="train" : statistics computed on the training part only (no leakage)  [default]
  fit_on="all"   : statistics computed on all data, as the paper's flowchart suggests
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FILES = {"coimbra": "coimbra.csv", "wdbc": "wdbc.csv", "wpbc": "wpbc.csv"}


def load_dataset(name, drop_features=()):
    """Return X (ndarray), y (ndarray, 1 = positive/malignant/recurrence/patient), feature names."""
    df = pd.read_csv(DATA_DIR / FILES[name])
    y = df["target"].to_numpy(dtype=int)
    X = df.drop(columns=["target"] + list(drop_features))
    return X.to_numpy(dtype=float), y, list(X.columns)


class Preprocessor:
    """Mean imputation, IQR outlier replacement by the mean, min-max scaling."""

    def __init__(self, outliers="iqr", iqr_k=1.5):
        self.outliers, self.iqr_k = outliers, iqr_k

    def fit(self, X):
        X = np.asarray(X, float)
        self.mean_ = np.nanmean(X, axis=0)
        Xi = np.where(np.isnan(X), self.mean_, X)
        if self.outliers == "iqr":
            q1, q3 = np.percentile(Xi, [25, 75], axis=0)
            self.lo_, self.hi_ = q1 - self.iqr_k * (q3 - q1), q3 + self.iqr_k * (q3 - q1)
            Xi = self._replace_outliers(Xi)
        self.min_, self.max_ = Xi.min(axis=0), Xi.max(axis=0)
        self.range_ = np.where(self.max_ - self.min_ == 0, 1.0, self.max_ - self.min_)
        return self

    def _replace_outliers(self, X):
        out = (X < self.lo_) | (X > self.hi_)
        return np.where(out, self.mean_, X)

    def transform(self, X):
        X = np.asarray(X, float)
        Xi = np.where(np.isnan(X), self.mean_, X)
        if self.outliers == "iqr":
            Xi = self._replace_outliers(Xi)
        return (Xi - self.min_) / self.range_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def split_and_preprocess(X, y, seed, test_size=0.2, fit_on="train", outliers="iqr"):
    """Stratified 80/20 split; returns Xtr, Xte, ytr, yte (preprocessed)."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    pre = Preprocessor(outliers=outliers)
    if fit_on == "all":
        pre.fit(X)
    elif fit_on == "train":
        pre.fit(Xtr)
    else:
        raise ValueError(f"fit_on must be 'train' or 'all', got {fit_on!r}")
    return pre.transform(Xtr), pre.transform(Xte), ytr, yte
