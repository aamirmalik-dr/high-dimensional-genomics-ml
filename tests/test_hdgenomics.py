import numpy as np
import pytest

from hdgenomics.classify import compare_models, cross_val_classification
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


def test_pca_orders_components_and_shape():
    data = synthetic_expression(n_samples=60, n_genes=800, n_informative=60, effect_size=2.0)
    xs, _ = standardize(data.X)
    emb, evr = pca_embedding(xs, 2)
    assert emb.shape == (60, 2)
    # Explained variance is sorted in descending order and lies in [0, 1].
    assert evr[0] >= evr[1]
    assert np.all(evr >= 0) and np.all(evr <= 1)


def test_pca_and_clustering_recover_structure():
    data = synthetic_expression(n_samples=60, n_genes=800, n_informative=60, effect_size=2.0)
    xs, _ = standardize(data.X)
    emb, _ = pca_embedding(xs, 2)
    clusters = kmeans_clusters(emb, 2)
    ari = cluster_agreement(data.y, clusters)
    assert ari > 0.5  # clustering should track the planted classes


def test_benjamini_hochberg_matches_known_values():
    # Reference values from R's p.adjust(method="BH") on the same input.
    p = np.array([0.005, 0.01, 0.03, 0.04, 0.5])
    q = benjamini_hochberg(p)
    expected = np.array([0.025, 0.025, 0.05, 0.05, 0.5])
    assert np.allclose(q, expected, atol=1e-9)


def test_benjamini_hochberg_preserves_input_order():
    # Same values, shuffled: q-values must follow their original position.
    p = np.array([0.03, 0.005, 0.5, 0.01, 0.04])
    q = benjamini_hochberg(p)
    expected = np.array([0.05, 0.025, 0.5, 0.025, 0.05])
    assert np.allclose(q, expected, atol=1e-9)


def test_benjamini_hochberg_monotone_and_bounded():
    p = np.array([0.001, 0.01, 0.03, 0.5, 0.9])
    q = benjamini_hochberg(p)
    assert np.all(q >= 0) and np.all(q <= 1)
    assert np.all(q >= p)  # adjusted p-values are never smaller than raw
    # Monotone in the ranked p-values.
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)


def test_differential_expression_recovers_signal():
    data = synthetic_expression(n_samples=60, n_genes=1000, n_informative=50, effect_size=2.0)
    de = differential_expression(data.X, data.y, data.gene_names)
    # Informative genes are named sig_* and placed first.
    sig_idx = [i for i, n in enumerate(data.gene_names) if n.startswith("sig_")]
    recovered = de.significant[sig_idx].mean()
    assert recovered > 0.5
    # Fold change direction matches the planted positive shift on class 1.
    assert de.log2_fold_change[sig_idx].mean() > 0


def test_differential_expression_requires_two_classes():
    X = np.random.default_rng(0).standard_normal((10, 5))
    y = np.zeros(10, dtype=int)
    with pytest.raises(ValueError):
        differential_expression(X, y, [f"g{i}" for i in range(5)])


@pytest.mark.parametrize("model", ["logreg", "svm", "rf"])
def test_classification_beats_chance(model):
    data = synthetic_expression(n_samples=60, n_genes=500, n_informative=50, effect_size=2.0)
    xs, _ = standardize(data.X)
    res = cross_val_classification(xs, data.y, model=model)
    assert res["accuracy_mean"] > 0.7
    assert 0.0 <= res["f1_macro_mean"] <= 1.0


def test_cross_val_rejects_unknown_model():
    data = synthetic_expression(n_samples=20, n_genes=30)
    with pytest.raises(ValueError):
        cross_val_classification(data.X, data.y, model="nope")


def test_compare_models_returns_all():
    data = synthetic_expression(n_samples=40, n_genes=300, n_informative=30, effect_size=2.0)
    xs, _ = standardize(data.X)
    results = compare_models(xs, data.y)
    assert set(results) == {"logreg", "svm", "rf"}
    for m in results.values():
        assert "accuracy_mean" in m and "f1_macro_mean" in m
