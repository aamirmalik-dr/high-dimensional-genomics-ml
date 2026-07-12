# Results

All numbers below were produced by `scripts/analyze.py` and `scripts/benchmark.py`
on the committed sample `data/leukemia_sample.csv` (72 samples, the top 1000 most
variable genes of the Golub leukemia set, classes ALL and AML). Seed 0, 5-fold
stratified cross-validation. The raw metrics are in `results/metrics.json`.

## Unsupervised structure

| Metric                              | Value |
|-------------------------------------|-------|
| PCA explained variance (PC1)        | 0.165 |
| PCA explained variance (PC2)        | 0.120 |
| K-means vs true labels (adjusted Rand) | 0.016 |

The adjusted Rand index is near zero. This is an honest negative result: the
largest axes of variation in the standardized expression are not the ALL versus
AML contrast, so k-means over the most variable genes does not recover the
diagnostic labels. See `results/pca.png`.

## Supervised classification

Cross-validated on the same standardized features and folds for every model.

| Model               | Accuracy        | Macro F1 |
|---------------------|-----------------|----------|
| Logistic regression | 0.958 +/- 0.057 | 0.955    |
| Linear SVM          | 0.944 +/- 0.053 | 0.940    |
| Random forest       | 0.986 +/- 0.029 | 0.984    |

All three models separate the subtypes well. The random forest edges out the
linear models here, though with 72 samples the fold-to-fold spread overlaps and
the difference should not be over-read.

## Differential expression

| Metric                          | Value        |
|---------------------------------|--------------|
| Genes tested                    | 1000         |
| Significant at FDR 0.05         | 362          |

Per-gene Welch t-tests with a from-scratch Benjamini-Hochberg FDR correction.
The volcano plot in `results/volcano.png` shows mean expression difference
(AML minus ALL) against `-log10` of the FDR q-value, with genes passing FDR 0.05
highlighted.

## Relationship to the full matrix

The committed sample is exactly the top 1000 most variable genes that the
pipeline selects from the full 7129-gene matrix, so the PCA, clustering, and
linear-classifier numbers above match a full-matrix run. Differential expression
on the full matrix flags 1056 significant genes of 7129 at FDR 0.05; the sample
reports a higher significant fraction (362 of 1000) because it is deliberately
enriched for the most variable, and therefore most discriminative, genes. Run
`scripts/download_data.py` then `scripts/analyze.py --csv data/leukemia.csv` to
reproduce the full-matrix figures.
