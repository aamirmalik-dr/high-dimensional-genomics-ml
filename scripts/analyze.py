"""End-to-end high-dimensional gene-expression analysis.

Runs the full pipeline on a committed expression matrix and writes figures and a
metrics file:

1. Keep the most variable genes and standardize.
2. PCA embedding and k-means clustering, scored against the true labels.
3. Cross-validated classification (logistic regression, linear SVM, random
   forest).
4. Per-gene differential expression with Benjamini-Hochberg FDR control.

By default it reads the small committed sample ``data/leukemia_sample.csv`` and
runs fully offline. Point ``--csv`` at the full matrix from
``scripts/download_data.py`` to reproduce the whole-genome numbers.

Usage:
    # Offline quickstart on the committed sample:
    python scripts/analyze.py

    # On the full Golub matrix:
    python scripts/download_data.py --out data/leukemia.csv
    python scripts/analyze.py --csv data/leukemia.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hdgenomics.classify import compare_models
from hdgenomics.data import load_leukemia, synthetic_expression
from hdgenomics.dimreduce import cluster_agreement, kmeans_clusters, pca_embedding
from hdgenomics.preprocess import standardize, top_variance_genes
from hdgenomics.stats import differential_expression

DEFAULT_SAMPLE = Path("data/leukemia_sample.csv")


def plot_pca(embedding: np.ndarray, y: np.ndarray, class_names: list[str], path: Path) -> None:
    """Write a 2D PCA scatter colored by class label."""
    plt.figure(figsize=(5, 4))
    for cls in np.unique(y):
        m = y == cls
        plt.scatter(embedding[m, 0], embedding[m, 1], label=class_names[cls], alpha=0.75, s=30)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of expression by subtype")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_volcano(
    log2_fc: np.ndarray, q_values: np.ndarray, significant: np.ndarray, alpha: float, path: Path
) -> None:
    """Write a volcano plot of fold change against FDR-adjusted significance."""
    neg_log_q = -np.log10(np.clip(q_values, 1e-300, 1.0))
    plt.figure(figsize=(6, 4.5))
    plt.scatter(
        log2_fc[~significant],
        neg_log_q[~significant],
        s=10,
        alpha=0.4,
        color="#9aa0a6",
        label="not significant",
    )
    plt.scatter(
        log2_fc[significant],
        neg_log_q[significant],
        s=14,
        alpha=0.7,
        color="#c62828",
        label=f"FDR <= {alpha:g}",
    )
    plt.axhline(-np.log10(alpha), color="#1a237e", linestyle="--", linewidth=1)
    plt.xlabel("mean expression difference (AML - ALL)")
    plt.ylabel("-log10(FDR q-value)")
    plt.title("Differential expression volcano plot")
    plt.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=None, help="expression CSV (default: committed sample)")
    parser.add_argument("--synthetic", action="store_true", help="use the synthetic generator")
    parser.add_argument("--out", default="results")
    parser.add_argument("--top-genes", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        data = synthetic_expression()
        source = "synthetic"
    else:
        csv_path = args.csv or DEFAULT_SAMPLE.as_posix()
        data = load_leukemia(csv_path)
        source = Path(csv_path).as_posix()
    print(
        f"Data: {data.n_samples} samples, {data.n_genes} genes, "
        f"classes={data.class_names} (source: {source})"
    )

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

    # Classification: compare a few models with the same folds and seed.
    models = compare_models(X_std, data.y)
    for name, m in models.items():
        print(
            f"{name:>7} CV accuracy: {m['accuracy_mean']:.3f} +/- {m['accuracy_std']:.3f} "
            f"(macro F1 {m['f1_macro_mean']:.3f}, {m['n_splits']} folds)"
        )

    # Differential expression on all genes.
    de = differential_expression(data.X, data.y, data.gene_names, alpha=args.alpha)
    n_sig = int(de.significant.sum())
    print(f"Differentially expressed genes at FDR {args.alpha:g}: {n_sig} of {data.n_genes}")

    # Figures.
    plot_pca(embedding, data.y, data.class_names, out_dir / "pca.png")
    plot_volcano(
        de.log2_fold_change, de.q_values, de.significant, args.alpha, out_dir / "volcano.png"
    )

    # Metrics file.
    metrics = {
        "source": source,
        "n_samples": data.n_samples,
        "n_genes": data.n_genes,
        "class_names": data.class_names,
        "top_genes_used": int(len(keep)),
        "pca_explained_variance": [float(evr[0]), float(evr[1])],
        "kmeans_adjusted_rand": float(ari),
        "models": models,
        "differential_expression": {
            "alpha": args.alpha,
            "n_significant": n_sig,
            "n_tested": data.n_genes,
        },
        "seed": 0,
    }
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"Wrote figures and {metrics_path.name} to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
