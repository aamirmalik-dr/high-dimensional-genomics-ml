"""Cross-validated model comparison on high-dimensional expression data.

Standardizes the most variable genes and cross-validates logistic regression, a
linear SVM, and a random forest with shared folds and a fixed seed, then prints a
comparison table. Runs offline on the committed sample by default.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --csv data/leukemia.csv --top-genes 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hdgenomics.classify import compare_models
from hdgenomics.data import load_leukemia, synthetic_expression
from hdgenomics.preprocess import standardize, top_variance_genes

DEFAULT_SAMPLE = Path("data/leukemia_sample.csv")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=None, help="expression CSV (default: committed sample)")
    parser.add_argument("--synthetic", action="store_true", help="use the synthetic generator")
    parser.add_argument("--top-genes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.synthetic:
        data = synthetic_expression()
        source = "synthetic"
    else:
        source = args.csv or str(DEFAULT_SAMPLE)
        data = load_leukemia(source)

    keep = top_variance_genes(data.X, args.top_genes)
    X_std, _ = standardize(data.X[:, keep])
    results = compare_models(X_std, data.y, seed=args.seed)

    n_folds = next(iter(results.values()))["n_splits"]
    print(
        f"Source: {source} ({data.n_samples} samples, {len(keep)} genes, "
        f"{n_folds}-fold CV, seed {args.seed})\n"
    )
    header = f"{'model':<10}{'accuracy':<20}{'macro F1':<12}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        acc = f"{m['accuracy_mean']:.3f} +/- {m['accuracy_std']:.3f}"
        print(f"{name:<10}{acc:<20}{m['f1_macro_mean']:<12.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
