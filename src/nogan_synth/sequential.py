"""Sequential (panel/longitudinal) synthesis: wraps either
`AutoregressiveSynthesizer` or `NoGANSynthesizer` -- both are otherwise
flat/i.i.d.-row methods -- to generate variable-length groups of rows
(``group_id`` + an implicit time order within each group) instead of
independent rows.

Design: a real sequence is (row_0, row_1, ..., row_{T-1}) for a group.
Row_0 has no history, so it's modeled like a flat dataset (an
`AutoregressiveSynthesizer`/`NoGANSynthesizer` fit on just the first row of
every group). Every row_t (t>0) is modeled as a function of row_{t-1}
("lag-1"): a second synthesizer is fit on a *transition frame* built by
pairing each row with its predecessor within the same group (lag columns
prefixed ``lag__``), so the same tree-conditioning / kernel-blend machinery
each synthesizer already has does the work -- this module only supplies the
lag columns as fixed, already-known predictors:

- `AutoregressiveSynthesizer.sample(given=lag_df)` (added in autoreg_synth.py)
  treats the lag columns as a given prefix of `order_` and only generates
  the rest via its existing per-column CART chain.
- `NoGANSynthesizer.sample(seeds=...)` (added in synthesizer.py) is instead
  driven by 1-nearest-neighbor lookup (in lag-column embedding space) from
  the given lag context to a real transition row, then blends from that
  row's own kernel window as usual -- "snap to the closest analogous real
  transition, then kernel-blend its outcome."

Sequence length is bootstrap-sampled from the real per-group row-count
distribution (no explicit length model); `group_id` is freshly generated
(random hex, not reused from real data) since it carries no distributional
signal of its own.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from .autoreg_synth import AutoregressiveSynthesizer, greedy_association_order
from .embeddings import resolve_embedding
from .synthesizer import NoGANSynthesizer

LAG_PREFIX = "lag__"


def _lag_cols(cols: list[str]) -> list[str]:
    return [f"{LAG_PREFIX}{c}" for c in cols]


def _initial_frame(df: pd.DataFrame, group_col: str, cols: list[str]) -> pd.DataFrame:
    return df.groupby(group_col, sort=False)[cols].first().reset_index(drop=True)


def _transition_frame(df: pd.DataFrame, group_col: str, cols: list[str]) -> pd.DataFrame:
    """One row per (t-1 -> t) real transition, lag columns + current columns."""
    prev = df.groupby(group_col, sort=False)[cols].shift(1)
    same_group = df[group_col] == df[group_col].shift(1)
    trans = pd.concat([prev.add_prefix(LAG_PREFIX), df[cols]], axis=1)
    return trans[same_group].reset_index(drop=True)


def _group_id(n: int, rng: np.random.Generator, width: int = 8) -> np.ndarray:
    ids = set()
    while len(ids) < n:
        ids.update(rng.bytes(4).hex() for _ in range(n - len(ids)))
    return np.array(list(ids)[:n])


class SequentialAutoregressiveSynthesizer(BaseEstimator):
    """`AutoregressiveSynthesizer`, group-and-lag aware. Same
    `min_samples_leaf`/`max_depth` knobs, applied to both the initial-row
    and transition models.
    """

    def __init__(
        self,
        group_col: str = "group_id",
        min_samples_leaf: int = 2,
        max_depth: int | None = None,
        random_state: int | None = None,
    ):
        self.group_col = group_col
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X: pd.DataFrame) -> "SequentialAutoregressiveSynthesizer":
        self.cols_ = [c for c in X.columns if c != self.group_col]
        self.group_sizes_ = X.groupby(self.group_col, sort=False).size().to_numpy()

        init_df = _initial_frame(X, self.group_col, self.cols_)
        self.init_model_ = AutoregressiveSynthesizer(
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            random_state=self.random_state,
        ).fit(init_df)

        trans_df = _transition_frame(X, self.group_col, self.cols_)
        lag_cols = _lag_cols(self.cols_)
        target_order = greedy_association_order(trans_df[self.cols_], self.cols_)
        self.trans_model_ = AutoregressiveSynthesizer(
            order=lag_cols + target_order,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            random_state=self.random_state,
        ).fit(trans_df[lag_cols + self.cols_])

        return self

    def sample(self, n_groups: int) -> pd.DataFrame:
        rng = np.random.default_rng(self.random_state)
        sizes = rng.choice(self.group_sizes_, size=n_groups)
        group_ids = _group_id(n_groups, rng)

        rows = self.init_model_.sample(n_groups)
        rows.insert(0, self.group_col, group_ids)
        rows.insert(1, "_t", 0)
        frames = [rows]

        active = np.arange(n_groups)
        cur = rows[self.cols_].reset_index(drop=True)
        t = 1
        while True:
            active = active[sizes[active] > t]
            if len(active) == 0:
                break
            given = cur.iloc[active].reset_index(drop=True).add_prefix(LAG_PREFIX)
            step = self.trans_model_.sample(given=given)
            step = step[self.cols_].reset_index(drop=True)
            out_step = step.copy()
            out_step.insert(0, self.group_col, group_ids[active])
            out_step.insert(1, "_t", t)
            frames.append(out_step)
            cur.iloc[active] = step.to_numpy()
            t += 1

        out = pd.concat(frames, ignore_index=True)
        out = out.sort_values([self.group_col, "_t"], kind="stable").drop(columns="_t")
        return out.reset_index(drop=True)[[self.group_col] + self.cols_]


class SequentialNoGANSynthesizer(BaseEstimator):
    """`NoGANSynthesizer`, group-and-lag aware -- same idea as
    `SequentialAutoregressiveSynthesizer` but each transition step snaps to
    the nearest real (lag -> current) example (1-NN in lag-embedding space)
    and kernel-blends from there, instead of tree-conditioning.
    """

    def __init__(
        self,
        group_col: str = "group_id",
        embedding: str | object = "onehot",
        jitter: float = 0.1,
        n_neighbors: int = 30,
        no_blend: list[str] = (),
        cat_resample: str = "copy",
        cat_swap_frac: float = 1.0,
        random_state: int | None = None,
    ):
        self.group_col = group_col
        self.embedding = embedding
        self.jitter = jitter
        self.n_neighbors = n_neighbors
        self.no_blend = no_blend
        self.cat_resample = cat_resample
        self.cat_swap_frac = cat_swap_frac
        self.random_state = random_state

    def fit(self, X: pd.DataFrame) -> "SequentialNoGANSynthesizer":
        self.cols_ = [c for c in X.columns if c != self.group_col]
        self.group_sizes_ = X.groupby(self.group_col, sort=False).size().to_numpy()

        init_df = _initial_frame(X, self.group_col, self.cols_)
        self.init_model_ = NoGANSynthesizer(
            embedding=self.embedding, jitter=self.jitter, n_neighbors=self.n_neighbors,
            no_blend=self.no_blend, cat_resample=self.cat_resample,
            cat_swap_frac=self.cat_swap_frac, random_state=self.random_state,
        ).fit(init_df)

        trans_df = _transition_frame(X, self.group_col, self.cols_)
        lag_cols = _lag_cols(self.cols_)
        self.trans_model_ = NoGANSynthesizer(
            embedding=self.embedding, jitter=self.jitter, n_neighbors=self.n_neighbors,
            no_blend=self.no_blend, cat_resample=self.cat_resample,
            cat_swap_frac=self.cat_swap_frac, random_state=self.random_state,
        ).fit(trans_df[lag_cols + self.cols_])

        self.lag_embedding_ = resolve_embedding(self.embedding)
        self.lag_embedding_.fit(trans_df[lag_cols])
        self.lag_train_z_ = self.lag_embedding_.transform(trans_df[lag_cols])
        self.lag_cols_ = lag_cols

        return self

    def _nearest_seeds(self, given_lag: pd.DataFrame) -> np.ndarray:
        from sklearn.neighbors import NearestNeighbors

        z = self.lag_embedding_.transform(given_lag)
        _, idx = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(self.lag_train_z_).kneighbors(z)
        return idx[:, 0]

    def sample(self, n_groups: int) -> pd.DataFrame:
        rng = np.random.default_rng(self.random_state)
        sizes = rng.choice(self.group_sizes_, size=n_groups)
        group_ids = _group_id(n_groups, rng)

        rows = self.init_model_.sample(n_groups)
        rows.insert(0, self.group_col, group_ids)
        rows.insert(1, "_t", 0)
        frames = [rows]

        active = np.arange(n_groups)
        cur = rows[self.cols_].reset_index(drop=True)
        t = 1
        while True:
            active = active[sizes[active] > t]
            if len(active) == 0:
                break
            given_lag = cur.iloc[active].reset_index(drop=True)
            given_lag.columns = self.lag_cols_
            seeds = self._nearest_seeds(given_lag)
            step = self.trans_model_.sample(seeds=seeds)
            step = step[self.cols_].reset_index(drop=True)
            out_step = step.copy()
            out_step.insert(0, self.group_col, group_ids[active])
            out_step.insert(1, "_t", t)
            frames.append(out_step)
            cur.iloc[active] = step.to_numpy()
            t += 1

        out = pd.concat(frames, ignore_index=True)
        out = out.sort_values([self.group_col, "_t"], kind="stable").drop(columns="_t")
        return out.reset_index(drop=True)[[self.group_col] + self.cols_]
