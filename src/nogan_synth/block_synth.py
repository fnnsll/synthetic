"""Block-tree kernel synthesizer: TreeKernelSynthesizer's Chow-Liu tree, but
with one dense, highly-interconnected group of columns (a "block") collapsed
into a single super-node instead of being torn apart into individual tree
edges.

TreeKernelSynthesizer alone measurably fails on data whose correlation
structure isn't tree-shaped -- this dataset's biggest cluster (~47 columns)
is a dense, nested blob (see association_matrix: no sharp threshold at
which it appears, it grows continuously as the threshold relaxes), not a
chain of pairwise dependencies. Approximating that blob with a spanning
tree conditions each of its columns on only one weak-in-isolation parent
edge, discarding the rest of the blob's joint structure -- measured as
badly as the copula mode this package already dropped (discriminator AUC
~0.999).

The fix: sample the block as one unit with NoGANSynthesizer (already shown
to reproduce that block's internal joint structure well -- it does a
global nearest-neighbor blend across the whole block at once, not a
column-by-column chain), then treat it as a single node in the outer tree.
Every other ("outside") column still gets a Chow-Liu tree built among
itself and the block. A column whose strongest real dependency is *to the
block* gets conditioned on the block's realized synthetic sub-row via
nearest-neighbor search in the block's own embedding space (not one
arbitrary column inside it); a column whose strongest dependency is to
another outside column chains off that column exactly like
TreeKernelSynthesizer.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors

from .synthesizer import NoGANSynthesizer, _infer_decimals
from .tree_synth import association_matrix, fit_conditioner

_BLOCK = "__BLOCK__"


class _BlockConditioner:
    """Picks a training row index per synthetic block sub-row, by nearest
    neighbor in the block's own (training-fit) embedding space -- so a
    satellite column conditions on the block's *whole realized state*,
    not one column inside it.
    """

    def __init__(self, block_synth: NoGANSynthesizer, n_neighbors: int):
        self.block_synth = block_synth
        Z_train = block_synth.Z_
        k = min(n_neighbors, len(Z_train))
        self.nn = NearestNeighbors(n_neighbors=k).fit(Z_train)
        median_dist = np.median(block_synth.neighbor_dist_[:, 1:]) or 1.0
        self.tau = 1.0 / (median_dist**2)

    def pick(
        self, block_rows: pd.DataFrame, rng: np.random.Generator, exclude: np.ndarray | None = None
    ) -> np.ndarray:
        Z_query = self.block_synth.embedding_.transform(block_rows)
        dist, idx = self.nn.kneighbors(Z_query)
        log_weights = -self.tau * dist**2
        if exclude is not None:
            log_weights = log_weights.copy()
            log_weights[idx == exclude[:, None]] = -np.inf
        gumbel = -np.log(-np.log(rng.uniform(size=log_weights.shape) + 1e-300) + 1e-300)
        local_choice = np.argmax(log_weights + gumbel, axis=1)
        return idx[np.arange(len(block_rows)), local_choice]


class BlockKernelSynthesizer(BaseEstimator):
    def __init__(
        self,
        block_cols: list[str],
        embedding: str | object = "onehot",
        block_n_neighbors: int = 50,
        block_jitter: float = 0.02,
        n_neighbors: int = 30,
        jitter: float = 0.05,
        min_association: float = 0.0,
        random_state: int | None = None,
    ):
        self.block_cols = block_cols
        self.embedding = embedding
        self.block_n_neighbors = block_n_neighbors
        self.block_jitter = block_jitter
        self.n_neighbors = n_neighbors
        self.jitter = jitter
        self.min_association = min_association
        self.random_state = random_state

    def fit(self, X: pd.DataFrame) -> "BlockKernelSynthesizer":
        self.X_ = X.reset_index(drop=True)
        self.block_cols_ = list(self.block_cols)
        self.outside_cols_ = [c for c in self.X_.columns if c not in self.block_cols_]
        self.cat_cols_ = set(self.X_.select_dtypes(include="object").columns)

        self.block_synth_ = NoGANSynthesizer(
            embedding=self.embedding,
            jitter=self.block_jitter,
            n_neighbors=self.block_n_neighbors,
            random_state=self.random_state,
        )
        self.block_synth_.fit(self.X_[self.block_cols_])

        if not self.outside_cols_:
            self.tree_ = nx.Graph()
            self.bfs_edges_ = []
            self._conditioners = {}
            return self

        assoc = association_matrix(self.X_, self.outside_cols_ + self.block_cols_)
        block_assoc = assoc.loc[self.outside_cols_, self.block_cols_].max(axis=1)

        graph = nx.Graph()
        graph.add_nodes_from([_BLOCK] + self.outside_cols_)
        for c in self.outside_cols_:
            graph.add_edge(_BLOCK, c, weight=max(block_assoc[c], 1e-6))
        for i, a in enumerate(self.outside_cols_):
            for b in self.outside_cols_[i + 1 :]:
                w = assoc.loc[a, b]
                graph.add_edge(
                    a, b, weight=max(w, 1e-6) if w > self.min_association else 1e-6
                )

        self.tree_ = nx.maximum_spanning_tree(graph, weight="weight")
        self.bfs_edges_ = list(nx.bfs_edges(self.tree_, _BLOCK))

        parent_cols = {parent for parent, _ in self.bfs_edges_ if parent != _BLOCK}
        self._conditioners = {
            parent: fit_conditioner(self.X_, parent, self.n_neighbors, self.cat_cols_)
            for parent in parent_cols
        }
        self._block_conditioner = _BlockConditioner(self.block_synth_, self.n_neighbors)

        num_cols = [c for c in self.outside_cols_ if c not in self.cat_cols_]
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

    def _pick(self, parent, query, rng, exclude=None):
        if parent == _BLOCK:
            return self._block_conditioner.pick(query, rng, exclude)
        return self._conditioners[parent].pick(query, rng, exclude)

    def sample(self, n_samples: int) -> pd.DataFrame:
        rng = np.random.default_rng(self.random_state)
        block_sample = self.block_synth_.sample(n_samples)

        if not self.outside_cols_:
            return block_sample[self.block_cols_]

        out = pd.DataFrame(index=range(n_samples), columns=self.outside_cols_, dtype=object)

        children_by_parent: dict[str, list[str]] = {}
        for parent, child in self.bfs_edges_:
            children_by_parent.setdefault(parent, []).append(child)

        # One shared neighbor-row draw per parent (including BLOCK itself),
        # reused for all its children -- see tree_synth.py's sample() for
        # why: independent per-child draws would pick a different real row
        # per sibling and destroy residual sibling correlation the parent
        # alone doesn't explain.
        for parent, children in children_by_parent.items():
            parent_query = block_sample if parent == _BLOCK else out[parent].to_numpy()
            primary_idx = self._pick(parent, parent_query, rng)
            if self.jitter:
                secondary_idx = self._pick(parent, parent_query, rng, exclude=primary_idx)
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

        return pd.concat([block_sample[self.block_cols_], out[self.outside_cols_]], axis=1)[
            self.X_.columns.tolist()
        ]
