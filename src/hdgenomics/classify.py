"""Supervised classification with cross-validation for wide data."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _make_pipeline(model: str, n_features: int) -> Pipeline:
    if model == "logreg":
        clf = LogisticRegression(C=1.0, max_iter=2000)
    elif model == "svm":
        clf = SVC(kernel="linear", C=1.0)
    else:
        raise ValueError(f"unknown model {model!r}; choose 'logreg' or 'svm'")
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def cross_val_classification(
    X: np.ndarray,
    y: np.ndarray,
    model: str = "logreg",
    n_splits: int = 5,
    seed: int = 0,
) -> dict[str, float]:
    """Cross-validated accuracy for a classifier on high-dimensional data.

    Standardization is fit inside each fold to avoid leakage. With very few
    samples, ``n_splits`` is reduced to the smallest class count.

    Args:
        X: Feature matrix.
        y: Class labels.
        model: ``"logreg"`` or ``"svm"``.
        n_splits: Requested number of folds.
        seed: Random seed for the fold shuffling.

    Returns:
        Mean and standard deviation of the fold accuracies.
    """
    min_class = int(np.min(np.bincount(y)))
    n_splits = max(2, min(n_splits, min_class))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accuracies: list[float] = []
    for train_idx, test_idx in skf.split(X, y):
        pipe = _make_pipeline(model, X.shape[1])
        pipe.fit(X[train_idx], y[train_idx])
        accuracies.append(float(pipe.score(X[test_idx], y[test_idx])))
    arr = np.asarray(accuracies)
    return {"accuracy_mean": float(arr.mean()), "accuracy_std": float(arr.std()), "n_splits": n_splits}
