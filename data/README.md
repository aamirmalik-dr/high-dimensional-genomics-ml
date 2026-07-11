# Data

This directory is gitignored. No datasets are committed.

## Golub leukemia (real)

`scripts/download_data.py` fetches the classic Golub et al. 1999 leukemia
microarray dataset from OpenML (72 samples, ~7000 genes, two classes ALL and
AML) and writes it as `data/leukemia.csv` with a `label` column:

```bash
python scripts/download_data.py --out data/leukemia.csv
```

If OpenML is unreachable, the script writes a reproducible synthetic
high-dimensional dataset instead and says so.

## Synthetic (offline)

The unit tests and the default `analyze.py` invocation need no download. They
build a synthetic two-class expression matrix (`synthetic_expression`) with a
known set of differentially expressed genes, so the differential-expression and
classification steps have a ground truth to recover. This keeps continuous
integration fully self-contained.
