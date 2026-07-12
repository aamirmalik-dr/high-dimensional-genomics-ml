# Data

## leukemia_sample.csv (committed)

`leukemia_sample.csv` is a small, license-clean carve-out of the public Golub et
al. 1999 leukemia microarray dataset. It contains all 72 samples but only the top
1000 most variable genes, plus a `label` column of ALL/AML. It is about 348 KB, so
it is committed to the repository and the offline demo runs with no download:

```bash
python scripts/analyze.py     # reads data/leukemia_sample.csv by default
python scripts/benchmark.py
```

This is a real public subset, not synthetic. It is a most-variable-gene subset,
so its differential-expression fraction is higher than the whole genome. Use the
full matrix below for the genome-wide number.

## leukemia.csv (full, not committed)

`scripts/download_data.py` fetches the full Golub matrix from OpenML (72 samples,
7129 genes, classes ALL and AML) and writes it as `data/leukemia.csv` with a
`label` column. The full file (about 2 MB) is gitignored:

```bash
python scripts/download_data.py --out data/leukemia.csv
python scripts/analyze.py --csv data/leukemia.csv
```

If OpenML is unreachable, the script writes a reproducible synthetic
high-dimensional dataset instead and says so.

## Synthetic (offline, for tests)

The unit tests build a synthetic two-class expression matrix
(`synthetic_expression`) with a known set of differentially expressed genes, so
the differential-expression and classification steps have a ground truth to
recover. This keeps continuous integration fully self-contained. Pass
`--synthetic` to `analyze.py` or `benchmark.py` to run on it directly.
