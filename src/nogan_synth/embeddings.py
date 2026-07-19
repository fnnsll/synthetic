"""Pluggable row embeddings for NoGANSynthesizer.

Each embedding exposes fit(df) / transform(df) -> np.ndarray so the
synthesizer's kernel distances can be computed in whatever space the
embedding defines. Any object with the same two methods can be passed
in directly instead of one of these.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def _split_dtypes(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols]
    return cat_cols, num_cols


class OneHotEmbedding:
    """One-hot categoricals + scaled numerics. Default: safest distance metric."""

    def fit(self, df: pd.DataFrame) -> "OneHotEmbedding":
        self.cat_cols_, self.num_cols_ = _split_dtypes(df)
        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.encoder_.fit(df[self.cat_cols_].fillna("iamna"))
        self.scaler_ = StandardScaler()
        self.scaler_.fit(df[self.num_cols_].fillna(0))
        # Which numeric columns are ever missing, so presence/absence
        # becomes an explicit feature the nearest-neighbor distance sees --
        # plain fillna(0) alone puts every missing row at the same spot in
        # feature space regardless of its neighbors' actual presence
        # pattern, so joint missingness (e.g. two columns that are almost
        # always missing together) doesn't get matched.
        self.null_cols_ = [c for c in self.num_cols_ if df[c].isna().any()]
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        cat = self.encoder_.transform(df[self.cat_cols_].fillna("iamna"))
        num = self.scaler_.transform(df[self.num_cols_].fillna(0))
        if self.null_cols_:
            null_mask = df[self.null_cols_].isna().to_numpy().astype(float)
            return np.hstack([cat, num, null_mask])
        return np.hstack([cat, num])


class LabelEmbedding:
    """Ordinal-encode categoricals + scale numerics. Cheapest, roughest distances."""

    def fit(self, df: pd.DataFrame) -> "LabelEmbedding":
        self.cat_cols_, self.num_cols_ = _split_dtypes(df)
        self.encoders_ = {}
        for col in self.cat_cols_:
            le = LabelEncoder()
            le.fit(df[col].fillna("iamna"))
            self.encoders_[col] = le
        self.scaler_ = StandardScaler()
        self.scaler_.fit(df[self.num_cols_].fillna(0))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        cat_arrs = []
        for col in self.cat_cols_:
            le = self.encoders_[col]
            filled = df[col].fillna("iamna")
            known = set(le.classes_)
            safe = filled.where(filled.isin(known), le.classes_[0])
            cat_arrs.append(le.transform(safe).reshape(-1, 1))
        cat = np.hstack(cat_arrs) if cat_arrs else np.empty((len(df), 0))
        num = self.scaler_.transform(df[self.num_cols_].fillna(0))
        return np.hstack([cat, num])

    def inverse_transform(self, Z: np.ndarray) -> pd.DataFrame:
        """Decode a (possibly interpolated, non-integer) label-space point back
        to raw columns: nearest valid category code per categorical column,
        inverse-scale the numeric block. Used to decode a point blended in
        embedding space back to something with real column values.
        """
        n_cat = len(self.cat_cols_)
        data = {}
        for i, col in enumerate(self.cat_cols_):
            le = self.encoders_[col]
            codes = np.clip(np.round(Z[:, i]).astype(int), 0, len(le.classes_) - 1)
            data[col] = le.inverse_transform(codes)
        num_vals = self.scaler_.inverse_transform(Z[:, n_cat:])
        for i, col in enumerate(self.num_cols_):
            data[col] = num_vals[:, i]
        return pd.DataFrame(data)[self.cat_cols_ + self.num_cols_]


class UMAPEmbedding:
    """Nonlinear embedding via umap.UMAP, for when linear distance is too crude.

    Supports inverse_transform so the synthesizer can blend two rows *in*
    the learned embedding space and decode the interpolated point back,
    instead of blending raw features -- meaningful here specifically
    because UMAP is nonlinear, so embedding-space interpolation actually
    differs from (and should stay closer to the data manifold than) a raw
    feature blend. For a linear embedding (e.g. Whitened) this would be a
    no-op: blending commutes with any linear map, so there'd be nothing to
    gain from doing it in transformed space.
    """

    def __init__(self, n_components: int = 10, n_neighbors: int = 20):
        self.n_components = n_components
        self.n_neighbors = n_neighbors

    def fit(self, df: pd.DataFrame) -> "UMAPEmbedding":
        import umap

        self._base = LabelEmbedding().fit(df)
        raw = self._base.transform(df)
        self.reducer_ = umap.UMAP(
            n_components=self.n_components, n_neighbors=self.n_neighbors
        )
        self.reducer_.fit(raw)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        raw = self._base.transform(df)
        return self.reducer_.transform(raw)

    def inverse_transform(self, Z: np.ndarray) -> pd.DataFrame:
        raw = self.reducer_.inverse_transform(Z)
        return self._base.inverse_transform(raw)


class Whitened:
    """Wraps a base embedding with a Mahalanobis-equivalent whitening transform:
    plain Euclidean distance on the whitened output equals Mahalanobis distance
    on the base embedding, using a shrinkage covariance estimate (Ledoit-Wolf)
    so it stays invertible even with collinear one-hot columns. This is the
    closest thing to "learning the metric" without needing labels -- it
    reweights/rotates axes so distance actually respects the data's
    correlation structure instead of treating every embedded dimension as
    independent, which is what plain Euclidean on raw one-hot/scaled features
    implicitly (and usually wrongly) assumes.
    """

    def __init__(self, base, max_condition: float = 100.0):
        self.base = base
        # Caps eigval_max / eigval_min. Without this, one-hot columns for
        # rare categories have near-zero variance, so their whitening weight
        # (eigval**-0.5) explodes and rare-category alignment ends up
        # dominating the distance metric over everything else.
        self.max_condition = max_condition

    def fit(self, df: pd.DataFrame) -> "Whitened":
        self.base.fit(df)
        Z = self.base.transform(df)
        self.mean_ = Z.mean(axis=0)
        cov = LedoitWolf().fit(Z - self.mean_).covariance_
        eigval, eigvec = np.linalg.eigh(cov)
        eigval = np.clip(eigval, eigval.max() / self.max_condition, None)
        self.W_ = eigvec @ np.diag(eigval**-0.5) @ eigvec.T
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        Z = self.base.transform(df)
        return (Z - self.mean_) @ self.W_


EMBEDDINGS = {
    "onehot": OneHotEmbedding,
    "label": LabelEmbedding,
    "umap": UMAPEmbedding,
    "onehot-whiten": lambda: Whitened(OneHotEmbedding()),
    "label-whiten": lambda: Whitened(LabelEmbedding()),
}


def resolve_embedding(embedding):
    if isinstance(embedding, str):
        return EMBEDDINGS[embedding]()
    return embedding
