"""Differential expression testing with multiple-hypothesis correction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class DifferentialExpressionResult:
    """Per-gene test statistics for a two-class comparison.

    Attributes:
        gene_names: Gene names aligned with the arrays below.
        log2_fold_change: Mean difference (class 1 minus class 0).
        p_values: Raw two-sided t-test p-values.
        q_values: Benjamini-Hochberg FDR-adjusted p-values.
        significant: Boolean mask of genes with ``q_values <= alpha``.
    """

    gene_names: list[str]
    log2_fold_change: np.ndarray
    p_values: np.ndarray
    q_values: np.ndarray
    significant: np.ndarray


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg FDR-adjusted p-values (q-values).

    The adjustment controls the expected proportion of false discoveries among
    the rejected hypotheses. Output q-values are clipped to ``[0, 1]`` and are
    monotone in the ranked p-values.
    """
    p = np.asarray(p_values, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    # Enforce monotonicity from the largest p-value downward.
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out


def differential_expression(
    X: np.ndarray, y: np.ndarray, gene_names: list[str], alpha: float = 0.05
) -> DifferentialExpressionResult:
    """Two-sample t-test per gene with Benjamini-Hochberg correction.

    Args:
        X: Expression matrix, shape ``(n_samples, n_genes)``.
        y: Binary class labels.
        gene_names: Names aligned with the columns of ``X``.
        alpha: FDR threshold for the significance mask.

    Returns:
        Per-gene fold changes, p-values, q-values, and a significance mask.

    Raises:
        ValueError: If ``y`` does not have exactly two classes.
    """
    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError("differential_expression requires exactly two classes")
    group0 = X[y == classes[0]]
    group1 = X[y == classes[1]]
    # Welch's t-test per gene (unequal variance).
    t_stat, p_values = stats.ttest_ind(group1, group0, axis=0, equal_var=False)
    p_values = np.nan_to_num(p_values, nan=1.0)
    fold_change = group1.mean(axis=0) - group0.mean(axis=0)
    q_values = benjamini_hochberg(p_values)
    significant = q_values <= alpha
    return DifferentialExpressionResult(
        gene_names=list(gene_names),
        log2_fold_change=fold_change,
        p_values=p_values,
        q_values=q_values,
        significant=significant,
    )
