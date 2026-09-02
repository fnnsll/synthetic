"""NoGAN kernel-memorization synthesizer: fit/sample, sklearn-style.

Generates synthetic rows by kernel-weighted resampling of real training
rows in an embedding space (Gaussian RBF weights, tau = bandwidth), then
blends each row's numeric columns toward a second kernel-drawn neighbor
with a single shared mixing weight (mixup-style interpolation, not
independent per-column noise -- independent per-column jitter breaks
the joint covariance between numeric columns in a way a discriminator
detects even at tiny magnitudes). Blended values are then re-quantized to
each column's observed decimal precision: this dataset's numeric columns
are rounded to 1-3 decimals, so any interpolation producing full
float64 precision is an instant, magnitude-independent tell -- a more
fundamental leak than the covariance one. No adversarial training, no
gradient descent -- the "model" is {X, embedding, tau} per the RBF
kernel-memorization method.

Categoricals default to a weighted pick among the n_mix blend members
(`cat_resample="copy"`), which at small jitter is ~always the primary
neighbor -- so a synthetic row's 20-column categorical tuple is a verbatim
copy of one training row's, and that is exactly nogan's DCR/NNDR failure
mode on the MOSTLY AI Prize eval. `cat_resample="kernel"` instead draws
each categorical column independently from the seed's kernel-weighted
neighbor window, so the tuple is a local recombination, not a copy. It
trades categorical joint fidelity for record-level privacy; pair it with
the marginal selector in nogan_synth.resample to recover the joint.

A Gaussian-copula numeric mode was tried and dropped: ~29 of this
dataset's 60 numeric columns are point-mass/near-binary (>30% of rows
tied to one value), and the copula's inverse-ECDF decode collapses joint
structure on those columns (measured: a 0.60 fit-time rank correlation
came back as 0.35 Pearson / 0.32 Spearman after decode) -- diffuse
across ~29 columns, not fixable by scoping to smaller correlation
clusters. Discriminator AUC: copula ~0.9999 (trivially separable) vs.
mixup's ~0.535 (near-ideal 0.5). See nogan_synth.evaluate for the
per_column_discriminator_importance diagnostic that traced this.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors

from .embeddings import resolve_embedding


def _infer_decimals(col: pd.Series, max_decimals: int = 6) -> int:
    """Minimal decimal precision that exactly reproduces every observed value."""
    vals = col.dropna().to_numpy()
    if len(vals) == 0:
        return max_decimals
    for d in range(max_decimals + 1):
        scaled = vals * (10**d)
        if np.allclose(scaled, np.round(scaled), atol=1e-6):
            return d
    return max_decimals


class NoGANSynthesizer(BaseEstimator):
    def __init__(
        self,
        embedding: str | object = "onehot",
        metric: str = "euclidean",
        tau: str | float = "auto",
        n_neighbors: int = 50,
        jitter: float = 0.2,
        n_mix: int = 2,
        match_marginals: bool | list[str] = False,
        no_blend: list[str] = (),
        cat_resample: str = "copy",
        cat_block_threshold: float = 0.15,
        cat_swap_frac: float = 1.0,
        random_state: int | None = None,
    ):
        self.embedding = embedding
        self.metric = metric
        self.tau = tau
        self.n_neighbors = n_neighbors
        self.jitter = jitter
        self.n_mix = n_mix
        self.match_marginals = match_marginals
        self.no_blend = no_blend
        self.cat_resample = cat_resample
        self.cat_block_threshold = cat_block_threshold
        self.cat_swap_frac = cat_swap_frac
        self.random_state = random_state

    def fit(self, X: pd.DataFrame) -> "NoGANSynthesizer":
        self.X_ = X.reset_index(drop=True)
        self.embedding_ = resolve_embedding(self.embedding)
        self.embedding_.fit(self.X_)
        Z = self.embedding_.transform(self.X_)
        self.Z_ = Z

        n_neighbors = min(self.n_neighbors, len(self.X_))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=self.metric)
        nn.fit(Z)
        self.neighbor_dist_, self.neighbor_idx_ = nn.kneighbors(Z)

        if self.tau == "auto":
            median_dist = np.median(self.neighbor_dist_[:, 1:]) or 1.0
            self.tau_ = 1.0 / (median_dist**2)
        else:
            self.tau_ = float(self.tau)

        num_cols = self.X_.select_dtypes(exclude="object").columns.tolist()
        self.num_cols_ = num_cols
        self.cat_cols_ = self.X_.select_dtypes(include="object").columns.tolist()
        self.cat_blocks_ = self._derive_cat_blocks()
        self.int_cols_mask_ = np.array(
            [pd.api.types.is_integer_dtype(self.X_[c]) for c in num_cols]
        )
        self.decimals_ = np.array(
            [
                0 if is_int else _infer_decimals(self.X_[c])
                for c, is_int in zip(num_cols, self.int_cols_mask_)
            ]
        )
        # Columns with any missing values can't be continuously blended
        # (1-lam)*primary + lam*secondary -- if either member is NaN the
        # arithmetic poisons the result to NaN even when the other member
        # had a perfectly good real value, silently inflating missingness
        # in the synthetic output. Those columns instead get a single
        # weighted-multinomial member pick (like categoricals) so a real
        # value from one member never gets NaN-contaminated by the other.
        # `no_blend` opts in more columns to that same single-value pick --
        # for wide-range columns, any convex blend shrinks variance toward
        # the center regardless of transform space (tested log and
        # Yeo-Johnson blending, both flat-to-worse); using one real member's
        # exact value instead trades away only the blend's marginal
        # diversity boost for that column, not its joint consistency (the
        # chosen value is still a real, jointly-consistent observation).
        no_blend_set = set(self.no_blend)
        self._dense_num_idx_ = [
            i
            for i, c in enumerate(num_cols)
            if not self.X_[c].isna().any() and c not in no_blend_set
        ]
        self._nullable_num_idx_ = [
            i
            for i, c in enumerate(num_cols)
            if self.X_[c].isna().any() or c in no_blend_set
        ]

        return self

    def _match_marginal_indices(self) -> list[int]:
        if self.match_marginals is True:
            return list(range(len(self.num_cols_)))
        if not self.match_marginals:
            return []
        name_to_idx = {c: i for i, c in enumerate(self.num_cols_)}
        return [name_to_idx[c] for c in self.match_marginals if c in name_to_idx]

    def _quantile_match(self, arr: np.ndarray, col_indices: list[int]) -> np.ndarray:
        """Remap the given numeric columns back onto the real training
        marginal, by rank within the synthetic batch -- fixes mixup's
        regression-to-the-mean blur on wide-range columns (blending shrinks
        variance and distorts shape) while staying a monotonic per-column
        transform, so each row's relative position (and thus most of the
        joint structure mixup built) survives. Doing this to every column
        measurably over-corrects (discriminator AUC 0.696 -> 0.933 on this
        dataset) -- scope `col_indices` to the specific columns that need it.
        """
        out = arr.copy()
        for j in col_indices:
            col = self.num_cols_[j]
            real_vals = self.X_[col].to_numpy()
            real_vals = real_vals[~np.isnan(real_vals)]
            sorted_real = np.sort(real_vals)
            grid = (np.arange(len(sorted_real)) + 0.5) / len(sorted_real)

            col_vals = arr[:, j]
            valid = ~np.isnan(col_vals)
            ranks = pd.Series(col_vals[valid]).rank(method="average", pct=True).to_numpy()
            out[valid, j] = np.interp(ranks, grid, sorted_real)
        return out

    def _assign_numeric(self, df: pd.DataFrame, arr: np.ndarray) -> pd.DataFrame:
        arr = arr.copy()
        for j, d in enumerate(self.decimals_):
            arr[:, j] = np.round(arr[:, j], int(d))
        df[self.num_cols_] = arr
        int_col_names = [
            c for c, is_int in zip(self.num_cols_, self.int_cols_mask_) if is_int
        ]
        if int_col_names:
            df[int_col_names] = df[int_col_names].astype(int)
        return df

    def _kernel_choice(
        self,
        seeds: np.ndarray,
        rng: np.random.Generator,
        exclude_local: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        dists = self.neighbor_dist_[seeds]
        log_weights = -self.tau_ * dists**2
        if exclude_local is not None:
            log_weights = log_weights.copy()
            log_weights[np.arange(len(seeds)), exclude_local] = -np.inf
        gumbel = -np.log(-np.log(rng.uniform(size=log_weights.shape) + 1e-300) + 1e-300)
        local_choice = np.argmax(log_weights + gumbel, axis=1)
        return local_choice, self.neighbor_idx_[seeds, local_choice]

    def _draw_members(self, seeds: np.ndarray, rng: np.random.Generator, n_mix: int) -> np.ndarray:
        """n_mix distinct kernel-drawn neighbors per row (sequential exclusion),
        column 0 is always the closest (primary) draw.
        """
        n = len(seeds)
        dists = self.neighbor_dist_[seeds]
        exclude_mask = np.zeros_like(dists, dtype=bool)
        members = np.empty((n, n_mix), dtype=int)
        for j in range(n_mix):
            log_weights = np.where(exclude_mask, -np.inf, -self.tau_ * dists**2)
            gumbel = -np.log(-np.log(rng.uniform(size=log_weights.shape) + 1e-300) + 1e-300)
            local_choice = np.argmax(log_weights + gumbel, axis=1)
            members[:, j] = self.neighbor_idx_[seeds, local_choice]
            exclude_mask[np.arange(n), local_choice] = True
        return members

    def _draw_weights(self, n_samples: int, n_mix: int, rng: np.random.Generator) -> np.ndarray:
        """Blend weights: primary keeps (1 - lam), the other n_mix - 1 members
        split lam via a Dirichlet draw -- generalizes the 2-member (1-lam, lam)
        split to a k-member weighted centroid, same one-shared-weight-vector-
        per-row design as the 2-member case (still one real neighbor set, not
        independent per-column noise).
        """
        if n_mix == 1:
            return np.ones((n_samples, 1))
        lam = rng.uniform(0.0, min(self.jitter, 1.0), size=n_samples)
        if n_mix == 2:
            other = lam[:, None]
        else:
            other = rng.dirichlet(np.ones(n_mix - 1), size=n_samples) * lam[:, None]
        return np.concatenate([(1 - lam)[:, None], other], axis=1)

    @staticmethod
    def _weighted_pick(stack: np.ndarray, weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """stack: (n, n_mix, n_features). Picks one member per (row, feature)
        via weighted multinomial on `weights` (n, n_mix) -- shared mechanism
        for categoricals and nullable numeric columns, both of which need a
        single real value rather than a continuous blend.
        """
        n_mix = weights.shape[1]
        cum_weights = np.cumsum(weights, axis=1)
        u = rng.uniform(size=(stack.shape[0], stack.shape[2]))
        choice_idx = (cum_weights[:, :, None] <= u[:, None, :]).sum(axis=1)
        choice_idx = np.clip(choice_idx, 0, n_mix - 1)
        return np.take_along_axis(stack, choice_idx[:, None, :], axis=1)[:, 0, :]

    def _derive_cat_blocks(self) -> list[list[str]]:
        """Group categorical columns into correlated blocks (association-graph
        connected components). `cat_resample="block"` draws one kernel neighbor
        per block, so the strong within-block joint is copied intact from a
        real row while the full tuple across blocks is a recombination -- the
        privacy win without the joint-destroying per-column independence.
        """
        import networkx as nx

        from .tree_synth import association_matrix

        cols = self.cat_cols_
        if len(cols) < 2:
            return [list(cols)]
        A = association_matrix(self.X_[cols], cols)
        G = nx.Graph()
        G.add_nodes_from(cols)
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                if A.loc[a, b] >= self.cat_block_threshold:
                    G.add_edge(a, b)
        return [sorted(c) for c in nx.connected_components(G)]

    def _kernel_cat_resample(
        self, seeds: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Kernel-weighted categorical draw from each seed's neighbor window,
        breaking the "whole tuple copied from one neighbor" pattern behind
        nogan's DCR/NNDR failure.

        - "kernel": one independent draw per (row, column). Maximum tuple
          novelty, but per-column independence wrecks the 20-way joint (a
          discriminator then separates trivially on this dense dataset).
        - "block": one draw per (row, correlated block). Within-block joint
          copied intact from a real row; cross-block combination is new.

        `cat_swap_frac` < 1 restricts the recombination to a weighted subset
        of rows instead of applying it everywhere (which is what pushes
        discriminator AUC to ~0.99 on this dataset -- see module docstring).
        The rest keep the primary neighbor's tuple verbatim (like
        `cat_resample="copy"`). Rows are picked by how interchangeable their
        top-2 kernel-weighted neighbors are (weight ratio close to 1): those
        are the "cheap" swaps -- the alternate neighbor is nearly as good a
        local match as the primary, so substituting it barely perturbs the
        marginal/joint fit while still breaking the exact-tuple copy.
        """
        dists = self.neighbor_dist_[seeds]
        log_w = -self.tau_ * dists**2
        win_idx = self.neighbor_idx_[seeds]
        rows = np.arange(len(seeds))
        primary_idx = win_idx[:, 0]
        col_pos = {c: j for j, c in enumerate(self.cat_cols_)}
        groups = (
            [[c] for c in self.cat_cols_]
            if self.cat_resample == "kernel"
            else self.cat_blocks_
        )

        swap_mask = np.ones(len(seeds), dtype=bool)
        if self.cat_swap_frac < 1.0 and log_w.shape[1] > 1:
            k = int(round(self.cat_swap_frac * len(seeds)))
            swap_mask = np.zeros(len(seeds), dtype=bool)
            if k > 0:
                top2 = np.partition(log_w, -2, axis=1)[:, -2:]
                ratio = top2[:, 0] - top2[:, 1]  # <=0; closer to 0 = interchangeable
                swap_mask[np.argpartition(ratio, -k)[-k:]] = True

        out = np.empty((len(seeds), len(self.cat_cols_)), dtype=object)
        for grp in groups:
            gumbel = -np.log(-np.log(rng.uniform(size=log_w.shape) + 1e-300) + 1e-300)
            picked = win_idx[rows, np.argmax(log_w + gumbel, axis=1)]
            chosen_idx = np.where(swap_mask, picked, primary_idx)
            for c in grp:
                out[:, col_pos[c]] = self.X_[c].to_numpy()[chosen_idx]
        return out

    def sample(self, n_samples: int | None = None, seeds: np.ndarray | None = None) -> pd.DataFrame:
        """Generate rows. If ``seeds`` is passed (indices into the training
        set), each output row is a kernel blend drawn from that seed row's
        own neighbor window instead of a randomly chosen one -- used by
        ``sequential.py`` to condition a step on "the real row nearest to
        the given lag context" rather than an unconditioned draw.
        """
        rng = np.random.default_rng(self.random_state)
        n_train = len(self.X_)
        n_mix = max(1, min(self.n_mix, len(self.neighbor_idx_[0])))

        if seeds is None:
            seeds = rng.integers(0, n_train, size=n_samples)
        else:
            n_samples = len(seeds)
        kernel_cats = (
            self._kernel_cat_resample(seeds, rng)
            if self.cat_resample in ("kernel", "block") and self.cat_cols_
            else None
        )

        if self.jitter and n_mix > 1:
            # Blend a shared-weight-vector centroid of n_mix kernel-drawn
            # neighbors (mixup-style; n_mix=2 is the original two-point
            # blend), rather than independent per-column noise -- keeps the
            # result on the joint manifold of jointly-consistent real rows
            # instead of scattering each column off it independently.
            # Categoricals get the same weights used as pick probabilities
            # (no continuous interpolation makes sense for them), so a
            # synthetic row's full column set isn't always a verbatim copy
            # of one single real row.
            members = self._draw_members(seeds, rng, n_mix)
            weights = self._draw_weights(n_samples, n_mix, rng)
            primary_idx = members[:, 0]
            out = self.X_.iloc[primary_idx].reset_index(drop=True)
            match_idx = self._match_marginal_indices()

            if hasattr(self.embedding_, "inverse_transform"):
                # Nonlinear embedding: interpolate in embedding space and
                # decode back, rather than blending raw columns -- for a
                # linear embedding these are the same operation, but for
                # something like UMAP the embedding-space centroid decodes
                # to a point that actually follows the learned manifold
                # instead of a raw-feature-space shortcut between real rows.
                Z_blend = (weights[:, :, None] * self.Z_[members]).sum(axis=1)
                decoded = self.embedding_.inverse_transform(Z_blend)
                out = decoded[self.X_.columns.tolist()]
                if self.num_cols_:
                    arr = out[self.num_cols_].to_numpy()
                    if match_idx:
                        arr = self._quantile_match(arr, match_idx)
                    out = self._assign_numeric(out, arr)

            elif self.num_cols_:
                num_stack = self.X_[self.num_cols_].to_numpy()[members]
                blended = np.empty((n_samples, len(self.num_cols_)))
                if self._dense_num_idx_:
                    dense_stack = num_stack[:, :, self._dense_num_idx_]
                    blended[:, self._dense_num_idx_] = (weights[:, :, None] * dense_stack).sum(
                        axis=1
                    )
                if self._nullable_num_idx_:
                    nullable_stack = num_stack[:, :, self._nullable_num_idx_]
                    blended[:, self._nullable_num_idx_] = self._weighted_pick(
                        nullable_stack, weights, rng
                    )
                if match_idx:
                    blended = self._quantile_match(blended, match_idx)
                out = self._assign_numeric(out, blended)

            if self.cat_cols_:
                if kernel_cats is not None:
                    out[self.cat_cols_] = kernel_cats
                else:
                    cat_stack = self.X_[self.cat_cols_].to_numpy()[members]
                    out[self.cat_cols_] = self._weighted_pick(cat_stack, weights, rng)

            return out

        _, primary_idx = self._kernel_choice(seeds, rng)
        out = self.X_.iloc[primary_idx].reset_index(drop=True)
        if kernel_cats is not None:
            out[self.cat_cols_] = kernel_cats
        return out
