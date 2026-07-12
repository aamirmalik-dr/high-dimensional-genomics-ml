"""Supervised classification with cross-validation for wide data."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODELS = ("logreg", "svm", "rf")


def _make_pipeline(model: str, seed: int = 0) -> Pipeline:
    """Build a standardize-then-classify pipeline for the named model.

    Args:
        model: One of ``"logreg"``, ``"svm"``, or ``"rf"``.
        seed: Random seed for the (randomized) estimators.

    Returns:
        An unfitted scikit-learn pipeline.

    Raises:
        ValueError: If ``model`` is not a known name.
    """
    if model == "logreg":
        clf = LogisticRegression(C=1.0, max_iter=2000)
    elif model == "svm":
        clf = SVC(kernel="linear", C=1.0)
    elif model == "rf":
        clf = RandomForestClassifier(n_estimators=300, random_state=seed)
    else:
        raise ValueError(f"unknown model {model!r}; choose one of {MODELS}")
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def cross_val_classification(
    X: np.ndarray,
    y: np.ndarray,
    model: str = "logreg",
    n_splits: int = 5,
    seed: int = 0,
) -> dict[str, float]:
    """Cross-validated accuracy and macro F1 for a classifier on wide data.

    Standardization is fit inside each fold to avoid leakage. With very few
    samples, ``n_splits`` is reduced to the smallest class count.

    Args:
        X: Feature matrix.
        y: Class labels.
        model: ``"logreg"``, ``"svm"``, or ``"rf"``.
        n_splits: Requested number of folds.
        seed: Random seed for the fold shuffling and estimators.

    Returns:
        Mean and standard deviation of fold accuracy, mean macro F1, and the
        number of folds actually used.
    """
    from sklearn.metrics import f1_score

    min_class = int(np.min(np.bincount(y)))
    n_splits = max(2, min(n_splits, min_class))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accuracies: list[float] = []
    f1s: list[float] = []
    for train_idx, test_idx in skf.split(X, y):
        pipe = _make_pipeline(model, seed=seed)
        pipe.fit(X[train_idx], y[train_idx])
        pred = pipe.predict(X[test_idx])
        accuracies.append(float((pred == y[test_idx]).mean()))
        f1s.append(float(f1_score(y[test_idx], pred, average="macro")))
    acc = np.asarray(accuracies)
    return {
        "accuracy_mean": float(acc.mean()),
        "accuracy_std": float(acc.std()),
        "f1_macro_mean": float(np.mean(f1s)),
        "n_splits": n_splits,
    }


def compare_models(
    X: np.ndarray,
    y: np.ndarray,
    models: tuple[str, ...] = MODELS,
    n_splits: int = 5,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Cross-validate several classifiers and return their metrics.

    Args:
        X: Feature matrix.
        y: Class labels.
        models: Model names to compare.
        n_splits: Requested number of cross-validation folds.
        seed: Random seed shared across models for a fair comparison.

    Returns:
        A mapping from model name to its metric dictionary.
    """
    return {
        m: cross_val_classification(X, y, model=m, n_splits=n_splits, seed=seed) for m in models
    }
