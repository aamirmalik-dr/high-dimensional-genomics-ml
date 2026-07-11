"""Dimensionality reduction and clustering for expression data."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def pca_embedding(
    X: np.ndarray, n_components: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Project samples onto their leading principal components.

    Args:
        X: Expression matrix, shape ``(n_samples, n_genes)``.
        n_components: Number of components to keep.

    Returns:
        A tuple ``(embedding, explained_variance_ratio)`` where ``embedding`` has
        shape ``(n_samples, n_components)``.
    """
    n_components = min(n_components, min(X.shape))
    pca = PCA(n_components=n_components, random_state=0)
    embedding = pca.fit_transform(X)
    return embedding, pca.explained_variance_ratio_


def kmeans_clusters(X: np.ndarray, n_clusters: int = 2, seed: int = 0) -> np.ndarray:
    """Cluster samples with k-means and return integer cluster labels."""
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    return km.fit_predict(X)


def cluster_agreement(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Adjusted Rand index between two labelings, in ``[-1, 1]``."""
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(labels_a, labels_b))
