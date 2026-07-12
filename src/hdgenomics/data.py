"""Data sources for high-dimensional gene-expression analysis.

Two sources are supported:

* :func:`load_leukemia` reads the public Golub leukemia microarray dataset (72
  samples, ~7000 genes, two classes ALL and AML) from a CSV fetched by
  ``scripts/download_data.py``.
* :func:`synthetic_expression` generates a reproducible high-dimensional
  two-class dataset with a known set of differentially expressed genes. This
  needs no download and is what the tests and CI use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ExpressionData:
    """A gene-expression matrix with sample labels and gene names.

    Attributes:
        X: Expression matrix, shape ``(n_samples, n_genes)``.
        y: Integer class labels, shape ``(n_samples,)``.
        gene_names: Names aligned with the columns of ``X``.
        class_names: Human-readable names for the integer labels.
    """

    X: np.ndarray
    y: np.ndarray
    gene_names: list[str]
    class_names: list[str]

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_genes(self) -> int:
        return self.X.shape[1]


def synthetic_expression(
    n_samples: int = 80,
    n_genes: int = 2000,
    n_informative: int = 40,
    effect_size: float = 1.5,
    seed: int = 0,
) -> ExpressionData:
    """Generate a two-class high-dimensional expression dataset.

    A random subset of ``n_informative`` genes is shifted between the two classes
    by ``effect_size`` standard deviations; the rest are pure noise. The
    informative genes are placed first so tests can check that differential
    expression recovers them.

    Args:
        n_samples: Total samples (split evenly between the two classes).
        n_genes: Number of genes (features).
        n_informative: Number of truly differential genes.
        effect_size: Mean shift for informative genes, in standard deviations.
        seed: Random seed.

    Returns:
        The generated dataset with informative genes named ``sig_*``.
    """
    rng = np.random.default_rng(seed)
    n_pos = n_samples // 2
    y = np.array([0] * (n_samples - n_pos) + [1] * n_pos)
    X = rng.standard_normal((n_samples, n_genes)).astype(np.float64)
    shift = np.zeros(n_genes)
    shift[:n_informative] = effect_size
    X[y == 1] += shift
    gene_names = [f"sig_{i}" if i < n_informative else f"gene_{i}" for i in range(n_genes)]
    return ExpressionData(X=X, y=y, gene_names=gene_names, class_names=["class_0", "class_1"])


def load_leukemia(csv_path: str) -> ExpressionData:
    """Load the Golub leukemia dataset from a CSV.

    The CSV is produced by ``scripts/download_data.py`` with genes as columns, a
    ``label`` column of ALL/AML, and the remaining columns numeric expression.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If no label column is found.
    """
    import csv

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [row for row in reader if row]

    label_idx = None
    for i, name in enumerate(header):
        if name.strip().lower() in {"label", "class", "target"}:
            label_idx = i
            break
    if label_idx is None:
        raise ValueError("no label/class/target column found in CSV")

    gene_cols = [i for i in range(len(header)) if i != label_idx]
    gene_names = [header[i] for i in gene_cols]
    labels_raw = [row[label_idx] for row in rows]
    classes = sorted(set(labels_raw))
    class_to_int = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_int[v] for v in labels_raw])
    X = np.array([[float(row[i]) for i in gene_cols] for row in rows], dtype=np.float64)
    return ExpressionData(X=X, y=y, gene_names=gene_names, class_names=classes)


def train_test_split_data(
    data: ExpressionData, test_fraction: float = 0.3, seed: int = 0
) -> tuple[ExpressionData, ExpressionData]:
    """Split samples into train and test sets, stratified by class."""
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for cls in np.unique(data.y):
        idx = np.where(data.y == cls)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(test_fraction * len(idx))))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    train = ExpressionData(data.X[train_idx], data.y[train_idx], data.gene_names, data.class_names)
    test = ExpressionData(data.X[test_idx], data.y[test_idx], data.gene_names, data.class_names)
    return train, test
