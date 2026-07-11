"""Preprocessing utilities for wide expression matrices."""

from __future__ import annotations

import numpy as np


def standardize(
    X_train: np.ndarray, X_test: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Standardize genes to zero mean and unit variance using train statistics.

    The test matrix, if given, is transformed with the training mean and scale so
    no information leaks from test to train.
    """
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    X_train_s = (X_train - mean) / std
    X_test_s = (X_test - mean) / std if X_test is not None else None
    return X_train_s, X_test_s


def top_variance_genes(X: np.ndarray, k: int) -> np.ndarray:
    """Return the column indices of the ``k`` highest-variance genes.

    A common first filter in high-dimensional expression analysis: keep the most
    variable genes and discard flat, uninformative ones.
    """
    if k >= X.shape[1]:
        return np.arange(X.shape[1])
    variances = X.var(axis=0)
    return np.argsort(variances)[::-1][:k]
