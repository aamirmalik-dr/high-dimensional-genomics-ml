"""Machine learning on high-dimensional gene-expression data.

A small toolkit for the classic "wide" biology setting where the number of genes
far exceeds the number of samples: dimensionality reduction and clustering,
supervised classification with cross-validation, and per-gene differential
expression testing with multiple-hypothesis correction.
"""

from hdgenomics.classify import compare_models, cross_val_classification
from hdgenomics.data import (
    ExpressionData,
    load_leukemia,
    synthetic_expression,
    train_test_split_data,
)
from hdgenomics.dimreduce import kmeans_clusters, pca_embedding
from hdgenomics.preprocess import standardize, top_variance_genes
from hdgenomics.stats import benjamini_hochberg, differential_expression

__all__ = [
    "ExpressionData",
    "load_leukemia",
    "synthetic_expression",
    "train_test_split_data",
    "pca_embedding",
    "kmeans_clusters",
    "standardize",
    "top_variance_genes",
    "benjamini_hochberg",
    "differential_expression",
    "cross_val_classification",
    "compare_models",
]

__version__ = "0.1.0"
