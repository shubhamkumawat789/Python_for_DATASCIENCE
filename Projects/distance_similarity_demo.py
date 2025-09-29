#!/usr/bin/env python3
"""
distance_similarity_demo.py

A beginner-friendly script to compute common distance and similarity measures
between samples (rows) in a dataset.

How to run:
    python distance_similarity_demo.py

What it does:
  1) Builds a small numeric dataset and (separately) a binary dataset
  2) Optionally standardizes numeric features
  3) Computes several pairwise distance/similarity matrices:
     - Euclidean distance
     - Manhattan (L1) distance
     - Chebyshev distance
     - Minkowski distance (p=3 by default)
     - Cosine distance & similarity
     - Correlation distance (1 - Pearson correlation)
     - Mahalanobis distance
     - Hamming distance (for binary features)
     - Jaccard distance & similarity (for binary features)

Why/When to use each:
  - Euclidean: default straight-line distance for continuous features (sensitive to scale).
  - Manhattan: sum of absolute differences; robust to outliers; useful with sparse/high-dim data.
  - Chebyshev: max absolute difference across features.
  - Minkowski: generalization; Euclidean (p=2), Manhattan (p=1) are special cases.
  - Cosine similarity: angle between vectors; good for text/tf-idf (magnitude-invariant).
  - Correlation distance: based on Pearson correlation; removes linear scaling/shift.
  - Mahalanobis: accounts for feature covariance; scale-invariant and de-correlated.
  - Hamming (binary): fraction of positions that differ.
  - Jaccard (binary): overlap / union over 1s; common for sets and multi-hot tags.

Notes:
  - Scaling matters! For distance on continuous features, consider standardization.
  - Handle missing values before computing distances (impute or drop).
  - Encode categoricals before using numeric distances (one-hot, ordinal if appropriate).
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.preprocessing import StandardScaler


def standardize(X: np.ndarray) -> np.ndarray:
    """Standardize features to zero mean and unit variance."""
    return StandardScaler().fit_transform(X)


def pairwise(X: np.ndarray, metric: str, **kwargs) -> pd.DataFrame:
    """
    Compute a pairwise distance matrix for rows of X using scipy.spatial.distance.cdist.
    Returns a pandas DataFrame for readability.
    """
    D = cdist(X, X, metric=metric, **kwargs)
    idx = [f"S{i+1}" for i in range(X.shape[0])]
    return pd.DataFrame(D, index=idx, columns=idx)


def cosine_similarity_matrix(X: np.ndarray) -> pd.DataFrame:
    """Compute cosine similarity = 1 - cosine distance."""
    D = cdist(X, X, metric="cosine")
    S = 1.0 - D
    idx = [f"S{i+1}" for i in range(X.shape[0])]
    return pd.DataFrame(S, index=idx, columns=idx)


def jaccard_binary_matrices(B: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For a binary matrix B (0/1):
        Jaccard distance d_J = 1 - (|A ∩ B| / |A ∪ B|) across 1s.
        Jaccard similarity s_J = 1 - d_J.
    """
    # SciPy's pdist/cdist use Jaccard distance for boolean arrays as defined above.
    # We'll use pdist -> squareform for a clean matrix.
    D = squareform(pdist(B.astype(bool), metric="jaccard"))
    S = 1.0 - D
    idx = [f"S{i+1}" for i in range(B.shape[0])]
    return (pd.DataFrame(D, index=idx, columns=idx),
            pd.DataFrame(S, index=idx, columns=idx))


def hamming_binary_matrix(B: np.ndarray) -> pd.DataFrame:
    """Hamming distance: fraction of positions that differ (for binary/categorical-encoded)."""
    D = squareform(pdist(B, metric="hamming"))
    idx = [f"S{i+1}" for i in range(B.shape[0])]
    return pd.DataFrame(D, index=idx, columns=idx)


def mahalanobis_matrix(X: np.ndarray) -> pd.DataFrame:
    """
    Compute Mahalanobis distance between rows using the inverse covariance (VI).
    If covariance is singular, add a small ridge for stability.
    """
    # Compute covariance across features (columns); shape (n_features, n_features)
    cov = np.cov(X, rowvar=False)
    # Regularize if near-singular
    eps = 1e-6
    cov_reg = cov + eps * np.eye(cov.shape[0])
    VI = np.linalg.inv(cov_reg)
    D = cdist(X, X, metric="mahalanobis", VI=VI)
    idx = [f"S{i+1}" for i in range(X.shape[0])]
    return pd.DataFrame(D, index=idx, columns=idx)


def correlation_distance_matrix(X: np.ndarray) -> pd.DataFrame:
    """
    Correlation distance = 1 - Pearson correlation between rows.
    SciPy's 'correlation' metric in cdist uses centered, normalized rows.
    """
    D = cdist(X, X, metric="correlation")
    idx = [f"S{i+1}" for i in range(X.shape[0])]
    return pd.DataFrame(D, index=idx, columns=idx)


def demo():
    # ---- Create a small numeric dataset (5 samples x 4 features) ----
    X = np.array([
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [6.2, 3.4, 5.4, 2.3],
        [5.9, 3.0, 5.1, 1.8],
        [5.5, 2.3, 4.0, 1.3],
    ], dtype=float)

    # Optional: standardized version of X (often better for distance comparisons)
    Xz = standardize(X)

    # ---- Create a small binary (multi-hot) dataset for Jaccard/Hamming ----
    # Imagine "skills" or "tags": 1 means present, 0 absent
    B = np.array([
        [1, 0, 1, 0, 0, 1],
        [1, 1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 0, 0, 1, 1],
    ], dtype=int)

    results = {}

    # --- Distances on standardized numeric features (recommended) ---
    results["Euclidean (std)"] = pairwise(Xz, "euclidean")
    results["Manhattan / L1 (std)"] = pairwise(Xz, "cityblock")
    results["Chebyshev (std)"] = pairwise(Xz, "chebyshev")
    results["Minkowski p=3 (std)"] = pairwise(Xz, "minkowski", p=3)
    results["Cosine distance (std)"] = pairwise(Xz, "cosine")
    results["Cosine similarity (std)"] = cosine_similarity_matrix(Xz)
    results["Correlation distance"] = correlation_distance_matrix(X)
    results["Mahalanobis distance"] = mahalanobis_matrix(X)

    # --- Binary distances/similarities ---
    jaccard_D, jaccard_S = jaccard_binary_matrices(B)
    results["Hamming distance (binary)"] = hamming_binary_matrix(B)
    results["Jaccard distance (binary)"] = jaccard_D
    results["Jaccard similarity (binary)"] = jaccard_S

    return results


if __name__ == "__main__":
    dfs = demo()
    # Print a compact summary to console
    for name, df in dfs.items():
        print("\\n=== " + name + " ===")
        print(df.round(3))
