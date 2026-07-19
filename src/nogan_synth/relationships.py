"""Detect columns whose values are the (near-)exact sum of other columns.

Useful before fitting NoGANSynthesizer: a sum-derived column (e.g. a total)
should be reconstructed from its parts after sampling rather than resampled
independently, or the synthetic rows will violate the real relationship.
"""
from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import pandas as pd


def _cluster_from_corr(corr: np.ndarray, threshold: float) -> list[list[int]]:
    """Connected components of columns linked by |correlation| > threshold."""
    graph = nx.Graph()
    graph.add_nodes_from(range(corr.shape[0]))
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if abs(corr[i, j]) > threshold:
                graph.add_edge(i, j)
    return [sorted(c) for c in nx.connected_components(graph)]


def correlation_clusters(
    df: pd.DataFrame, cols: list[str] | None = None, threshold: float = 0.15
) -> list[list[str]]:
    """Group numeric columns into correlation clusters, to scope the sum search."""
    if cols is None:
        cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[cols].corr().to_numpy()
    idx_clusters = _cluster_from_corr(corr, threshold)
    return [[cols[i] for i in cluster] for cluster in idx_clusters]


def check_sum_relationship(
    df: pd.DataFrame, target: str, parts: list[str], tol: float = 1e-6
) -> dict:
    """How well df[target] matches sum(df[parts]) row-wise.

    Returns match_frac (share of rows within tol) and residual stats over
    the rows that don't match, so a near-miss (rounding) is distinguishable
    from unrelated columns.
    """
    resid = df[target].to_numpy() - df[parts].sum(axis=1).to_numpy()
    match = np.abs(resid) <= tol
    return {
        "target": target,
        "parts": tuple(parts),
        "match_frac": float(match.mean()),
        "resid_mean_abs": float(np.abs(resid).mean()),
        "resid_max_abs": float(np.abs(resid).max()),
    }


def find_sum_relationships(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    max_parts: int = 2,
    tol: float = 1e-6,
    min_match_frac: float = 0.98,
    sample: int | None = 5000,
    cluster_threshold: float | None = 0.15,
) -> pd.DataFrame:
    """Search numeric columns for target = sum(parts) relationships.

    A sum relationship implies correlation, so by default the search is
    scoped to one correlation cluster at a time rather than every
    combination across all columns -- keeps the combinatorics down on wide
    data and avoids flagging coincidental cross-cluster sums. Pass
    `cluster_threshold=None` to search the full `cols` set unscoped.
    Row-sampled by default since exact matches don't need the full table.
    """
    if cols is None:
        cols = df.select_dtypes(include=np.number).columns.tolist()
    data = df[cols]
    if sample is not None and len(data) > sample:
        data = data.sample(n=sample, random_state=0)

    if cluster_threshold is None:
        groups = [cols]
    else:
        groups = correlation_clusters(data, cols, threshold=cluster_threshold)

    found = []
    for group in groups:
        for target in group:
            others = [c for c in group if c != target]
            for r in range(2, max_parts + 1):
                for parts in itertools.combinations(others, r):
                    result = check_sum_relationship(data, target, list(parts), tol=tol)
                    if result["match_frac"] >= min_match_frac:
                        found.append(result)

    result_df = pd.DataFrame(found)
    if result_df.empty:
        return result_df
    return result_df.sort_values(
        ["match_frac", "resid_mean_abs"], ascending=[False, True]
    ).reset_index(drop=True)
