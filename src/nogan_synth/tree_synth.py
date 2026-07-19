"""Chow-Liu-tree kernel synthesizer.

Sweeping a correlation-distance filtration (d = 1 - association) upward
from 0 and watching connected components merge is single-linkage
clustering -- which is exactly Kruskal's algorithm building a maximum-
spanning-tree by edge weight (= association strength). The H0 persistence
merge order of that filtration *is* this tree. Framed that way: every
column's single strongest edge determines its conditioning parent, with
no threshold and therefore no boundary. Contrast with a hard correlation-
cluster cut (nogan_synth.synthesizer's dropped copula mode, and the old
notebook pipeline's connected-components clustering) -- a threshold forces
a boundary column into one side or the other and models the seam with
zero cross-cluster dependency. A tree never has that seam.

Sampling walks the tree from a root (highest total association -- the
most globally connected column) in breadth-first order. The root is drawn
by a plain kernel resample of real rows. Every other column is drawn by a
*local* kernel step: given its parent's already-sampled value, find real
training rows with a similar parent value (nearest-neighbor + Gaussian RBF
weights for a numeric parent, exact category match for a categorical
parent), Gumbel-max sample one, and take that row's value for this column
(optionally jitter-blended toward a second such neighbor, same
duplicate-avoidance idea as NoGANSynthesizer). The joint is therefore a
chain of pairwise real matches along the tree, not one row copied
wholesale end to end -- the standard Chow-Liu tree-of-conditionals
approximation to the full joint, provably the best tree-structured
approximation in KL divergence.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats.contingency import association
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .reweighting import median_bandwidth
from .synthesizer import _infer_decimals


def _correlation_ratio(categories: np.ndarray, values: np.ndarray) -> float:
    """eta: share of numeric `values`' variance explained by `categories` groups."""
    df = pd.DataFrame({"cat": categories, "val": values}).dropna()
    if df["cat"].nunique() < 2 or len(df) < 2:
        return 0.0
    grand_mean = df["val"].mean()
    ss_total = ((df["val"] - grand_mean) ** 2).sum()
    if ss_total == 0:
        return 0.0
    ss_between = df.groupby("cat")["val"].apply(
        lambda g: len(g) * (g.mean() - grand_mean) ** 2
    ).sum()
    return float(np.sqrt(max(ss_between / ss_total, 0.0)))


def association_matrix(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Symmetric [0, 1] association strength between every pair of columns --
    |Pearson| for numeric-numeric, correlation ratio for numeric-categorical,
    Cramer's V for categorical-categorical. This is the edge weight the tree
    synthesizer's maximum-spanning-tree is built from.
    """
    cols = cols or df.columns.tolist()
    is_cat = {c: df[c].dtype == object for c in cols}
    mat = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            if is_cat[a] and is_cat[b]:
                ct = pd.crosstab(df[a], df[b])
                val = association(ct.to_numpy(), method="cramer") if ct.size else 0.0
            elif is_cat[a]:
                val = _correlation_ratio(df[a].to_numpy(), df[b].to_numpy())
            elif is_cat[b]:
                val = _correlation_ratio(df[b].to_numpy(), df[a].to_numpy())
            else:
                val = abs(df[[a, b]].corr().iloc[0, 1])
            val = 0.0 if pd.isna(val) else float(val)
            mat.loc[a, b] = mat.loc[b, a] = val
    return mat


class _CategoricalConditioner:
    """Picks a training row index per query row by exact category match."""

    def __init__(self, X: pd.DataFrame, col: str):
        self.groups = {cat: np.asarray(idx) for cat, idx in X.groupby(col).groups.items()}
        self.all_idx = X.index.to_numpy()

    def pick(
        self, query_vals: np.ndarray, rng: np.random.Generator, exclude: np.ndarray | None = None
    ) -> np.ndarray:
        picks = np.empty(len(query_vals), dtype=int)
        for i, cat in enumerate(query_vals):
            pool = self.groups.get(cat, self.all_idx)
            if exclude is not None and len(pool) > 1:
                pool = pool[pool != exclude[i]]
            picks[i] = pool[rng.integers(0, len(pool))]
        return picks


class _NumericConditioner:
    """Gumbel-max weighted pick of a training row index per query row, from
    the k nearest training rows by 1-D value distance (Gaussian RBF weights,
    median-heuristic bandwidth). NaN queries fall back to a NaN-valued pool.
    """

    def __init__(self, X: pd.DataFrame, col: str, n_neighbors: int):
        vals = X[[col]].fillna(X[col].median()).to_numpy()
        self.scaler = StandardScaler().fit(vals)
        encoded = self.scaler.transform(vals).ravel()
        k = min(n_neighbors, len(X))
        self.nn = NearestNeighbors(n_neighbors=k).fit(encoded.reshape(-1, 1))
        bw = median_bandwidth(encoded.reshape(-1, 1))
        self.tau = 1.0 / (bw**2) if bw else 1.0
        self.nan_pool = X.index[X[col].isna()].to_numpy()

    def pick(
        self, query_vals: np.ndarray, rng: np.random.Generator, exclude: np.ndarray | None = None
    ) -> np.ndarray:
        query_vals = np.asarray(query_vals, dtype=float)
        picks = np.empty(len(query_vals), dtype=int)

        nan_mask = np.isnan(query_vals)
        if nan_mask.any():
            pool = self.nan_pool if len(self.nan_pool) else np.arange(self.nn.n_samples_fit_)
            for i in np.where(nan_mask)[0]:
                cand = pool
                if exclude is not None and len(cand) > 1:
                    cand = cand[cand != exclude[i]]
                picks[i] = cand[rng.integers(0, len(cand))]

        valid = np.where(~nan_mask)[0]
        if len(valid):
            q = self.scaler.transform(query_vals[valid].reshape(-1, 1)).ravel()
            dist, idx = self.nn.kneighbors(q.reshape(-1, 1))
            log_weights = -self.tau * dist**2
            if exclude is not None:
                log_weights = log_weights.copy()
                log_weights[idx == exclude[valid, None]] = -np.inf
            gumbel = -np.log(-np.log(rng.uniform(size=log_weights.shape) + 1e-300) + 1e-300)
            local_choice = np.argmax(log_weights + gumbel, axis=1)
            picks[valid] = idx[np.arange(len(valid)), local_choice]

        return picks


def fit_conditioner(X: pd.DataFrame, col: str, n_neighbors: int, cat_cols: set[str]):
    if col in cat_cols:
        return _CategoricalConditioner(X, col)
    return _NumericConditioner(X, col, n_neighbors)


class TreeKernelSynthesizer(BaseEstimator):
    def __init__(
        self,
        n_neighbors: int = 50,
        jitter: float = 0.05,
        min_association: float = 0.0,
        random_state: int | None = None,
    ):
        self.n_neighbors = n_neighbors
        self.jitter = jitter
        self.min_association = min_association
        self.random_state = random_state

    def fit(self, X: pd.DataFrame) -> "TreeKernelSynthesizer":
        self.X_ = X.reset_index(drop=True)
        self.cols_ = list(X.columns)
        self.cat_cols_ = set(X.select_dtypes(include="object").columns)

        assoc = association_matrix(self.X_, self.cols_)
        graph = nx.Graph()
        graph.add_nodes_from(self.cols_)
        for i, a in enumerate(self.cols_):
            for b in self.cols_[i + 1 :]:
                w = assoc.loc[a, b]
                # Keep the graph complete (a tiny floor weight for near-zero
                # association) so maximum_spanning_tree always connects
                # every column -- an "independent" column still gets the
                # least-bad available parent rather than being dropped.
                graph.add_edge(a, b, weight=max(w, 1e-6) if w > self.min_association else 1e-6)
        self.association_ = assoc
        self.tree_ = nx.maximum_spanning_tree(graph, weight="weight")

        self.root_ = assoc.sum(axis=1).idxmax()
        self.bfs_edges_ = list(nx.bfs_edges(self.tree_, self.root_))

        parent_cols = {parent for parent, _ in self.bfs_edges_}
        self._conditioners = {
            parent: fit_conditioner(self.X_, parent, self.n_neighbors, self.cat_cols_)
            for parent in parent_cols
        }

        num_cols = [c for c in self.cols_ if c not in self.cat_cols_]
        self.num_cols_ = num_cols
        self.int_cols_mask_ = np.array(
            [pd.api.types.is_integer_dtype(self.X_[c]) for c in num_cols]
        )
        self.decimals_ = np.array(
            [
                0 if is_int else _infer_decimals(self.X_[c])
                for c, is_int in zip(num_cols, self.int_cols_mask_)
            ]
        )
        return self

    def _kernel_pick(
        self,
        parent: str,
        query_vals: np.ndarray,
        rng: np.random.Generator,
        exclude: np.ndarray | None = None,
    ) -> np.ndarray:
        return self._conditioners[parent].pick(query_vals, rng, exclude)

    def sample(self, n_samples: int) -> pd.DataFrame:
        rng = np.random.default_rng(self.random_state)
        n_train = len(self.X_)

        out = pd.DataFrame(index=range(n_samples), columns=self.cols_, dtype=object)
        root_idx = rng.integers(0, n_train, size=n_samples)
        out[self.root_] = self.X_[self.root_].to_numpy()[root_idx]

        children_by_parent: dict[str, list[str]] = {}
        for parent, child in self.bfs_edges_:
            children_by_parent.setdefault(parent, []).append(child)

        # One shared neighbor-row draw (and shared jitter weight) per parent,
        # reused for every one of its children -- not an independent draw
        # per child, which would pick a *different* real row per sibling
        # column and destroy exactly the residual correlation between
        # siblings that isn't already explained by the parent alone (same
        # reason NoGANSynthesizer's mixup uses one shared lambda per row).
        for parent, children in children_by_parent.items():
            parent_vals = out[parent].to_numpy()
            primary_idx = self._kernel_pick(parent, parent_vals, rng)
            if self.jitter:
                secondary_idx = self._kernel_pick(parent, parent_vals, rng, exclude=primary_idx)
                lam = rng.uniform(0.0, min(self.jitter, 1.0), size=n_samples)

            for child in children:
                child_vals = self.X_[child].to_numpy()[primary_idx]
                if self.jitter:
                    secondary_vals = self.X_[child].to_numpy()[secondary_idx]
                    if child in self.cat_cols_:
                        swap = rng.uniform(size=n_samples) < lam
                        child_vals = np.where(swap, secondary_vals, child_vals)
                    else:
                        child_vals = (1 - lam) * child_vals.astype(
                            float
                        ) + lam * secondary_vals.astype(float)
                out[child] = child_vals

        for j, col in enumerate(self.num_cols_):
            d = int(self.decimals_[j])
            out[col] = np.round(out[col].astype(float), d)
            if self.int_cols_mask_[j]:
                out[col] = out[col].astype(int)

        return out[self.cols_]
