"""Download the public Golub leukemia microarray dataset.

Fetches the dataset from OpenML (the classic Golub et al. 1999 leukemia set,
72 samples across ~7000 genes, two classes ALL and AML) and writes it to a CSV
with a ``label`` column. If OpenML is unreachable, writes a reproducible
synthetic high-dimensional dataset instead and says so.

Usage:
    python scripts/download_data.py --out data/leukemia.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from hdgenomics.data import synthetic_expression

OPENML_NAME = "leukemia"


def try_openml(out_path: Path) -> bool:
    """Fetch the leukemia dataset from OpenML and write a CSV. Return success."""
    try:
        from sklearn.datasets import fetch_openml

        bunch = fetch_openml(name=OPENML_NAME, version=1, as_frame=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  OpenML fetch failed: {exc}")
        return False

    frame = bunch.frame.copy()
    target_col = bunch.target.name
    frame = frame.rename(columns={target_col: "label"})
    frame.to_csv(out_path, index=False)
    print(f"Downloaded leukemia from OpenML ({frame.shape[0]} samples, "
          f"{frame.shape[1] - 1} genes) -> {out_path}")
    return True


def write_synthetic(out_path: Path) -> None:
    """Write a reproducible synthetic high-dimensional expression CSV."""
    data = synthetic_expression()
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(data.gene_names) + ["label"])
        for i in range(data.n_samples):
            row = [f"{v:.5f}" for v in data.X[i]]
            row.append(data.class_names[data.y[i]])
            writer.writerow(row)
    print(f"Wrote synthetic fallback dataset ({data.n_samples} samples, "
          f"{data.n_genes} genes) -> {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/leukemia.csv")
    args = parser.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not try_openml(out_path):
        print("Falling back to synthetic high-dimensional dataset.")
        write_synthetic(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
