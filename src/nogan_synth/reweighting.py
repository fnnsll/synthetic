"""Kernel Mean Matching (Huang et al., 2006) for post-hoc reweighting of
synthetic samples toward a real target distribution.

Matching marginals or pairs one at a time (IPF/raking-style) only
guarantees those lower-order projections line up -- it says nothing about
the full joint. KMM instead reweights synthetic rows so their *joint*
kernel mean embedding (over whatever column set you feed it) matches the
real data's, in one shot, no per-variable iteration. Feed it three weak
columns together and the weights correct the trivariate joint directly,
which is exactly the case marginal-by-marginal raking is weakest on.

Solved as a bound-constrained QP (0 <= weight <= B, weights sum ~= n) via
L-BFGS-B with a closed-form gradient -- no QP solver dependency needed;
the sum-to-n equality is relaxed to a quadratic penalty so plain
box-constrained L-BFGS-B applies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, minimize
from scipy.spatial.distance import cdist

from .embeddings import LabelEmbedding


def median_bandwidth(X: np.ndarray, sample: int = 2000, random_state: int = 0) -> float:
    """Median-heuristic Gaussian kernel bandwidth: median pairwise distance."""
    if len(X) > sample:
        rng = np.random.default_rng(random_state)
        X = X[rng.choice(len(X), size=sample, replace=False)]
    dists = cdist(X, X)
    return float(np.median(dists[dists > 0])) or 1.0


def gaussian_kernel(A: np.ndarray, B: np.ndarray, bandwidth: float) -> np.ndarray:
    sq_dists = cdist(A, B, metric="sqeuclidean")
    return np.exp(-sq_dists / (2 * bandwidth**2))


def kernel_mean_match(
    source: np.ndarray,
    target: np.ndarray,
    weight_cap: float = 1000.0,
    sum_penalty: float = 1.0,
    bandwidth: float | None = None,
) -> np.ndarray:
    """Weights for `source` rows so their weighted mean embedding matches
    `target`'s mean embedding in the same Gaussian-RKHS. Weights are
    bounded to [0, weight_cap] and softly pulled toward summing to
    len(source) (a uniform-weight baseline) via `sum_penalty`.
    """
    n, m = len(source), len(target)
    if bandwidth is None:
        bandwidth = median_bandwidth(np.vstack([source, target]))

    K = gaussian_kernel(source, source, bandwidth)
    kappa = (n / m) * gaussian_kernel(source, target, bandwidth).sum(axis=1)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        Kb = K @ beta
        excess = beta.sum() - n
        value = 0.5 * beta @ Kb - kappa @ beta + 0.5 * sum_penalty * excess**2
        grad = Kb - kappa + sum_penalty * excess
        return value, grad

    result = minimize(
        objective,
        x0=np.ones(n),
        jac=True,
        method="L-BFGS-B",
        bounds=Bounds(0.0, weight_cap),
    )
    return result.x


def _nystrom_features(
    X: np.ndarray, landmarks: np.ndarray, bandwidth: float, eps: float = 1e-8
) -> np.ndarray:
    """Explicit low-rank feature map Phi such that Phi @ Phi.T ~ K(X, X),
    built from the kernel to/between a small landmark set only (n x L
    instead of n x n).
    """
    K_mm = gaussian_kernel(landmarks, landmarks, bandwidth)
    K_mm[np.diag_indices_from(K_mm)] += eps
    eigval, eigvec = np.linalg.eigh(K_mm)
    eigval = np.clip(eigval, eps, None)
    inv_sqrt = eigvec @ np.diag(eigval**-0.5) @ eigvec.T
    K_nm = gaussian_kernel(X, landmarks, bandwidth)
    return K_nm @ inv_sqrt


def kernel_mean_match_nystrom(
    source: np.ndarray,
    target: np.ndarray,
    n_landmarks: int = 500,
    weight_cap: float = 1000.0,
    sum_penalty: float = 1.0,
    bandwidth: float | None = None,
    random_state: int = 0,
) -> np.ndarray:
    """Same objective as kernel_mean_match, but K is never formed explicitly
    -- every n x n product is replaced by two n x L products through a
    Nystrom low-rank feature map, so this scales to n in the tens of
    thousands where the exact QP's O(n^2) kernel matrix doesn't fit memory.
    """
    n, m = len(source), len(target)
    pool = np.vstack([source, target])
    if bandwidth is None:
        bandwidth = median_bandwidth(pool)

    n_landmarks = min(n_landmarks, len(pool))
    rng = np.random.default_rng(random_state)
    landmarks = pool[rng.choice(len(pool), size=n_landmarks, replace=False)]

    Phi_source = _nystrom_features(source, landmarks, bandwidth)
    Phi_target = _nystrom_features(target, landmarks, bandwidth)
    target_sum = Phi_target.sum(axis=0)
    kappa = (n / m) * (Phi_source @ target_sum)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        Kb = Phi_source @ (Phi_source.T @ beta)
        excess = beta.sum() - n
        value = 0.5 * beta @ Kb - kappa @ beta + 0.5 * sum_penalty * excess**2
        grad = Kb - kappa + sum_penalty * excess
        return value, grad

    result = minimize(
        objective,
        x0=np.ones(n),
        jac=True,
        method="L-BFGS-B",
        bounds=Bounds(0.0, weight_cap),
    )
    return result.x


def joint_kmm_weights(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    cols: list[str],
    embedding=None,
    **kmm_kwargs,
) -> np.ndarray:
    """KMM weights for `synthetic` rows matching the joint distribution of
    `real` over `cols` -- pass e.g. a weak triple of columns to correct
    their trivariate relationship directly, rather than one variable/pair
    at a time.
    """
    embedding = embedding or LabelEmbedding()
    embedding.fit(pd.concat([real[cols], synthetic[cols]], ignore_index=True))
    Z_syn = embedding.transform(synthetic[cols])
    Z_real = embedding.transform(real[cols])
    return kernel_mean_match(Z_syn, Z_real, **kmm_kwargs)


def joint_kmm_weights_nystrom(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    cols: list[str] | None = None,
    embedding=None,
    **kmm_kwargs,
) -> np.ndarray:
    """Nystrom version of joint_kmm_weights, for matching the full joint
    distribution (cols=None -> every column) at sample sizes where the
    exact n x n KMM QP doesn't fit memory -- catches diffuse joint mismatch
    spread thinly across many weakly-linked columns, not just one flagged
    triple.
    """
    if cols is None:
        cols = list(real.columns)
    embedding = embedding or LabelEmbedding()
    embedding.fit(pd.concat([real[cols], synthetic[cols]], ignore_index=True))
    Z_syn = embedding.transform(synthetic[cols])
    Z_real = embedding.transform(real[cols])
    return kernel_mean_match_nystrom(Z_syn, Z_real, **kmm_kwargs)


def weighted_resample(
    df: pd.DataFrame, weights: np.ndarray, n: int | None = None, random_state: int | None = None
) -> pd.DataFrame:
    """Bootstrap-resample `df` rows with probability proportional to `weights`."""
    n = n or len(df)
    probs = weights / weights.sum()
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(df), size=n, p=probs, replace=True)
    return df.iloc[idx].reset_index(drop=True)
