# High-dimensional genomics ML

Machine learning for the classic "wide" biology setting, where the number of
genes far exceeds the number of samples. The toolkit covers three tasks on the
same gene-expression matrix:

- **Dimensionality reduction and clustering**: PCA embeddings and k-means.
- **Supervised classification**: cross-validated logistic regression and a
  linear SVM, with standardization fit inside each fold to avoid leakage.
- **Differential expression**: a two-sample Welch t-test per gene with
  Benjamini-Hochberg FDR correction, and a volcano plot.

The Benjamini-Hochberg procedure is implemented from scratch in `stats.py` and
unit tested for monotonicity and bounds.

## What it does

- Loads the public Golub leukemia microarray dataset (ALL vs AML) or a
  reproducible synthetic dataset with a known set of differential genes.
- Filters to the most variable genes, standardizes, and runs the full pipeline.
- Reports PCA explained variance, cluster agreement, cross-validated accuracy,
  and the number of genes significant at a chosen FDR.

## What it does not do

- It does not perform batch-effect correction or gene-set enrichment.
- The models are standard linear classifiers, chosen for interpretability on
  small, wide data rather than maximum accuracy.
- Only two-class comparisons are implemented for differential expression.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Run

Fully offline (synthetic data with ground-truth signal genes):

```bash
python scripts/analyze.py
```

On the real Golub leukemia dataset:

```bash
python scripts/download_data.py --out data/leukemia.csv
python scripts/analyze.py --csv data/leukemia.csv
```

## Results

Measured on the real Golub leukemia dataset fetched from OpenML (72 samples,
7129 genes, classes ALL and AML), 5-fold stratified cross-validation, seed 0.
Produced by `scripts/analyze.py` in this repository.

| Step                                 | Result |
|--------------------------------------|--------|
| PCA explained variance (PC1, PC2)    | 0.165, 0.120 |
| Logistic regression CV accuracy      | 0.958 +/- 0.057 |
| Linear SVM CV accuracy               | 0.944 +/- 0.053 |
| Differential genes at FDR 0.05       | 1056 of 7129 |
| K-means vs true labels (adjusted Rand)| 0.016 |

The supervised classifiers separate ALL from AML very well (about 95 percent
accuracy), and roughly 15 percent of genes are differentially expressed at a 5
percent false discovery rate. The low adjusted Rand index is an honest negative
result: unsupervised k-means over the most variable genes does not recover the
diagnostic split, because the dominant axes of variation in the raw expression
are not the ALL/AML contrast. Supervised signal is there, but it is not the
largest source of variance.

On the synthetic dataset (which is engineered so the class contrast is a strong
signal), clustering does recover the planted labels; that path is used by the
tests to check the pipeline against a ground truth.

## Layout

```
src/hdgenomics/   data, preprocess, dimreduce, classify, stats
scripts/          download_data.py, analyze.py
notebooks/        demo.ipynb (executed)
tests/            pytest suite incl. FDR and signal-recovery checks
data/             gitignored; see data/README.md
```

## Tests

```bash
pytest -q
ruff check src tests scripts
```

## License

MIT, see [LICENSE](LICENSE).

## Author

Aamir Malik. [GitHub](https://github.com/aamirmalik-dr) ·
[LinkedIn](https://linkedin.com/in/dr-aamirmalik)
