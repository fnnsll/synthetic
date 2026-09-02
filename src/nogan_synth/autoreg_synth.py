"""CART-based autoregressive synthesizer (sequential conditional synthesis,
the standard `synthpop`-style method).

Every prior synthesizer this session either restricted conditioning to one
edge (TreeKernelSynthesizer, BlockKernelSynthesizer -- a Chow-Liu tree
provably can't represent this dataset's dense correlation blob) or blended
continuous values (NoGANSynthesizer's mixup, which shrinks variance on
wide-range columns) or assumed a parametric marginal shape (the dropped
copula mode, which broke on point-mass columns). This sidesteps all three:

Columns are ordered (greedily, by association strength -- most globally
connected first, then whichever remaining column is most associated with
the columns already placed). For each column after the first, a decision
tree is fit predicting it from *every* already-placed column (full
autoregressive conditioning, not one parent edge). At sample time, a
synthetic row's already-generated values are run through that tree to find
its leaf, and one real training row that landed in the same leaf is picked
at random -- the column's value comes from that real row. So every output
value is a real observed value (like NoGANSynthesizer, no blending, no
distributional assumption), but the *conditioning* is a full decision-tree
partition on the whole prefix instead of a single kernel-distance parent,
which is what a dense multi-column blob actually needs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .tree_synth import association_matrix


def greedy_association_order(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Order ``cols`` by association strength (most globally connected
    first, then whichever remaining column is most associated with the
    columns already placed) -- the same ordering `AutoregressiveSynthesizer`
    uses for its own columns, exposed standalone so `sequential.py` can order
    a target-column subset independent of any lag/exogenous columns.
    """
    assoc = association_matrix(df, cols)
    remaining = set(cols)
    first = assoc.sum(axis=1).idxmax()
    order = [first]
    remaining.remove(first)
    while remaining:
        best = max(remaining, key=lambda c: assoc.loc[c, order].max())
        order.append(best)
        remaining.remove(best)
    return order


class AutoregressiveSynthesizer(BaseEstimator):
    def __init__(
        self,
        order: list[str] | None = None,
        min_samples_leaf: int = 2,
        max_depth: int | None = None,
        random_state: int | None = None,
    ):
        self.order = order
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state

    def _greedy_order(self) -> list[str]:
        return greedy_association_order(self.X_, self.cols_)

    def _fit_predictor_encoder(self, col: str):
        if col in self.cat_cols_:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            enc.fit(self.X_[[col]].fillna("__NA__"))
            return ("cat", enc, None)
        vals = self.X_[[col]].to_numpy()
        null_mask = np.isnan(vals)
        fill_value = 0.0 if null_mask.all() else np.nanmedian(vals)
        filled = np.where(null_mask, fill_value, vals)
        scaler = StandardScaler().fit(filled)
        return ("num", scaler, fill_value)

    def _transform_predictor(self, col: str, series: pd.Series) -> np.ndarray:
        kind, enc, fill_value = self._encoders_[col]
        if kind == "cat":
            return enc.transform(series.fillna("__NA__").to_frame())
        vals = series.to_numpy(dtype=float).reshape(-1, 1)
        null_mask = np.isnan(vals)
        filled = np.where(null_mask, fill_value, vals)
        scaled = enc.transform(filled)
        return np.hstack([scaled, null_mask.astype(float)])

    def fit(self, X: pd.DataFrame) -> "AutoregressiveSynthesizer":
        self.X_ = X.reset_index(drop=True)
        self.cols_ = list(X.columns)
        self.cat_cols_ = set(X.select_dtypes(include="object").columns)

        self.order_ = list(self.order) if self.order is not None else self._greedy_order()

        self._encoders_ = {c: self._fit_predictor_encoder(c) for c in self.cols_}

        self.models_: dict[str, object] = {}
        self.leaf_pools_: dict[str, dict[int, np.ndarray]] = {}
        predictor_blocks: dict[str, np.ndarray] = {}

        for i, col in enumerate(self.order_[1:], start=1):
            predictors = self.order_[:i]
            for p in predictors:
                if p not in predictor_blocks:
                    predictor_blocks[p] = self._transform_predictor(p, self.X_[p])
            Xp = np.hstack([predictor_blocks[p] for p in predictors])

            if col in self.cat_cols_:
                y = self.X_[col].fillna("__NA__")
                tree = DecisionTreeClassifier(
                    min_samples_leaf=self.min_samples_leaf,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                )
            else:
                y = self.X_[col].fillna(self.X_[col].median())
                tree = DecisionTreeRegressor(
                    min_samples_leaf=self.min_samples_leaf,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                )
            tree.fit(Xp, y)

            leaves = tree.apply(Xp)
            self.models_[col] = tree
            self.leaf_pools_[col] = {
                leaf_id: np.where(leaves == leaf_id)[0] for leaf_id in np.unique(leaves)
            }

        num_cols = [c for c in self.cols_ if c not in self.cat_cols_]
        self.num_cols_ = num_cols
        self.int_cols_mask_ = np.array(
            [pd.api.types.is_integer_dtype(self.X_[c]) for c in num_cols]
        )
        return self

    def sample(self, n_samples: int | None = None, given: pd.DataFrame | None = None) -> pd.DataFrame:
        """Generate rows. If ``given`` is passed, its columns (a prefix of
        ``order_`` -- typically lag-1 features fixed by an outside caller,
        see ``sequential.py``) are used as-is instead of generated, and only
        used as predictors for the remaining columns. ``n_samples`` is then
        taken from ``len(given)``.
        """
        rng = np.random.default_rng(self.random_state)
        n_train = len(self.X_)
        n_samples = len(given) if given is not None else n_samples
        given_cols = set(given.columns) if given is not None else set()

        out = pd.DataFrame(index=range(n_samples), columns=self.cols_, dtype=object)
        for c in given_cols:
            out[c] = given[c].to_numpy()

        first = self.order_[0]
        if first not in given_cols:
            seed_idx = rng.integers(0, n_train, size=n_samples)
            out[first] = self.X_[first].to_numpy()[seed_idx]

        for i, col in enumerate(self.order_[1:], start=1):
            if col in given_cols:
                continue
            predictors = self.order_[:i]
            Xp = np.hstack([self._transform_predictor(p, out[p]) for p in predictors])
            tree = self.models_[col]
            leaves = tree.apply(Xp)
            pools = self.leaf_pools_[col]

            picks = np.empty(n_samples, dtype=int)
            for j, leaf_id in enumerate(leaves):
                pool = pools[leaf_id]
                picks[j] = pool[rng.integers(0, len(pool))]
            out[col] = self.X_[col].to_numpy()[picks]

        return out[self.cols_]
