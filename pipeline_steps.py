"""
Custom sklearn transformers shared between Main.py (training) and app/main.py (inference).
Keeping them in a dedicated module ensures joblib can resolve the class path when
unpickling the saved pipeline — it looks for e.g. pipeline_steps.NaNDropper, not __main__.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NaNDropper(BaseEstimator, TransformerMixin):
    """Drop features with any NaN/Inf or zero variance (fitted on train only)."""

    def fit(self, X, y=None):
        Xdf = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        bad = Xdf.replace([np.inf, -np.inf], np.nan).isna().any() | (Xdf.std() == 0)
        self.cols_to_keep_ = Xdf.columns[~bad].tolist()
        return self

    def transform(self, X):
        Xdf = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return Xdf[self.cols_to_keep_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.cols_to_keep_)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip feature values to ±n_std of the training distribution (fitted on train only)."""

    def __init__(self, n_std=2):
        self.n_std = n_std

    def fit(self, X, y=None):
        Xdf = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self.mean_ = Xdf.mean()
        self.std_  = Xdf.std()
        return self

    def transform(self, X):
        Xdf = (
            pd.DataFrame(X, columns=self.mean_.index)
            if not isinstance(X, pd.DataFrame) else X
        )
        return Xdf.clip(
            lower=self.mean_ - self.n_std * self.std_,
            upper=self.mean_ + self.n_std * self.std_,
            axis=1,
        )

    def get_feature_names_out(self, input_features=None):
        return np.array(self.mean_.index.tolist())


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop features where |Pearson r| > threshold with any earlier feature (fitted on train only)."""

    def __init__(self, threshold=0.90):
        self.threshold = threshold

    def fit(self, X, y=None):
        Xdf = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        corr  = Xdf.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        drop  = {c for c in upper.columns if upper[c].gt(self.threshold).any()}
        self.cols_to_keep_ = [c for c in Xdf.columns if c not in drop]
        self.n_dropped_    = len(drop)
        return self

    def transform(self, X):
        Xdf = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return Xdf[self.cols_to_keep_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.cols_to_keep_)
