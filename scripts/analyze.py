"""Run the full high-dimensional gene-expression analysis and write figures.

Pipeline: load data (real leukemia CSV or synthetic), PCA + k-means clustering,
cross-validated classification, and differential expression with FDR control.

Usage:
    # Fully offline synthetic run:
    python scripts/analyze.py

    # On a downloaded dataset:
    python scripts/analyze.py --csv data/leukemia.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hdgenomics.classify import cross_val_classification
from hdgenomics.data import load_leukemia, synthetic_expression
from hdgenomics.dimreduce import cluster_agreement, kmeans_clusters, pca_embedding
from hdgenomics.preprocess import standardize, top_variance_genes
from hdgenomics.stats import differential_expression


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=None, help="optional expression CSV")
    parser.add_argument("--out", default="results")
    parser.add_argument("--top-genes", type=int, default=1000)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_leukemia(args.csv) if args.csv else synthetic_expression()
    print(f"Data: {data.n_samples} samples, {data.n_genes} genes, "
          f"classes={data.class_names}")

    # Keep the most variable genes, then standardize.
    keep = top_variance_genes(data.X, args.top_genes)
    X = data.X[:, keep]
    X_std, _ = standardize(X)

    # PCA + clustering.
    embedding, evr = pca_embedding(X_std, n_components=2)
    clusters = kmeans_clusters(embedding, n_clusters=2)
    ari = cluster_agreement(data.y, clusters)
    print(f"PCA explained variance (PC1, PC2): {evr[0]:.3f}, {evr[1]:.3f}")
    print(f"K-means vs true labels adjusted Rand index: {ari:.3f}")

    # Classification.
    logreg = cross_val_classification(X_std, data.y, model="logreg")
    svm = cross_val_classification(X_std, data.y, model="svm")
    print(f"Logistic regression CV accuracy: {logreg['accuracy_mean']:.3f} "
          f"+/- {logreg['accuracy_std']:.3f} ({logreg['n_splits']} folds)")
    print(f"Linear SVM CV accuracy: {svm['accuracy_mean']:.3f} "
          f"+/- {svm['accuracy_std']:.3f} ({svm['n_splits']} folds)")

    # Differential expression on all genes.
    de = differential_expression(data.X, data.y, data.gene_names)
    n_sig = int(de.significant.sum())
    print(f"Differentially expressed genes at FDR 0.05: {n_sig} of {data.n_genes}")

    # Figures.
    plt.figure(figsize=(5, 4))
    for cls in np.unique(data.y):
        m = data.y == cls
        plt.scatter(embedding[m, 0], embedding[m, 1], label=data.class_names[cls], alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of expression")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "pca.png", dpi=120)
    plt.close()

    plt.figure(figsize=(5, 4))
    neg_log_q = -np.log10(np.clip(de.q_values, 1e-300, 1.0))
    plt.scatter(de.log2_fold_change, neg_log_q, s=8, alpha=0.5)
    plt.axhline(-np.log10(0.05), color="red", linestyle="--", linewidth=1)
    plt.xlabel("mean difference (class 1 - class 0)")
    plt.ylabel("-log10(FDR q-value)")
    plt.title("Volcano plot")
    plt.tight_layout()
    plt.savefig(out_dir / "volcano.png", dpi=120)
    plt.close()

    print(f"Wrote figures to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
