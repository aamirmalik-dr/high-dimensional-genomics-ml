import numpy as np

from hdgenomics.classify import cross_val_classification
from hdgenomics.data import (
    synthetic_expression,
    train_test_split_data,
)
from hdgenomics.dimreduce import cluster_agreement, kmeans_clusters, pca_embedding
from hdgenomics.preprocess import standardize, top_variance_genes
from hdgenomics.stats import benjamini_hochberg, differential_expression


def test_synthetic_shapes():
    data = synthetic_expression(n_samples=40, n_genes=500, n_informative=20)
    assert data.X.shape == (40, 500)
    assert set(np.unique(data.y)) == {0, 1}
    assert data.n_genes == 500


def test_train_test_split_stratified():
    data = synthetic_expression(n_samples=40, n_genes=100)
    train, test = train_test_split_data(data, test_fraction=0.25, seed=0)
    assert train.n_samples + test.n_samples == data.n_samples
    # Both classes present in both splits.
    assert set(np.unique(train.y)) == {0, 1}
    assert set(np.unique(test.y)) == {0, 1}


def test_standardize_no_leak():
    data = synthetic_expression(n_samples=30, n_genes=50)
    tr, te = train_test_split_data(data, seed=1)
    xs, xt = standardize(tr.X, te.X)
    assert np.allclose(xs.mean(axis=0), 0, atol=1e-6)
    assert xt.shape == te.X.shape


def test_top_variance_genes():
    X = np.array([[0.0, 0.0, 0.0], [0.0, 5.0, 1.0], [0.0, 10.0, 2.0]])
    idx = top_variance_genes(X, 2)
    assert 1 in idx  # highest variance column
    assert 0 not in idx  # zero-variance column excluded


def test_pca_and_clustering_recover_structure():
    data = synthetic_expression(n_samples=60, n_genes=800, n_informative=60, effect_size=2.0)
    xs, _ = standardize(data.X)
    emb, evr = pca_embedding(xs, 2)
    assert emb.shape == (60, 2)
    assert evr[0] >= evr[1]
    clusters = kmeans_clusters(emb, 2)
    ari = cluster_agreement(data.y, clusters)
    assert ari > 0.5  # clustering should track the planted classes


def test_benjamini_hochberg_monotone_and_bounded():
    p = np.array([0.001, 0.01, 0.03, 0.5, 0.9])
    q = benjamini_hochberg(p)
    assert np.all(q >= 0) and np.all(q <= 1)
    assert np.all(q >= p)  # adjusted p-values are never smaller than raw


def test_differential_expression_recovers_signal():
    data = synthetic_expression(n_samples=60, n_genes=1000, n_informative=50, effect_size=2.0)
    de = differential_expression(data.X, data.y, data.gene_names)
    # Informative genes are named sig_* and placed first.
    sig_idx = [i for i, n in enumerate(data.gene_names) if n.startswith("sig_")]
    recovered = de.significant[sig_idx].mean()
    assert recovered > 0.5


def test_classification_beats_chance():
    data = synthetic_expression(n_samples=60, n_genes=500, n_informative=50, effect_size=2.0)
    xs, _ = standardize(data.X)
    res = cross_val_classification(xs, data.y, model="logreg")
    assert res["accuracy_mean"] > 0.7
