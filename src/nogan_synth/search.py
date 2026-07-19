"""Multi-split robustness search for NoGANSynthesizer hyperparameters.

A single train/holdout split can't tell memorization apart from real
generalization, since both halves are drawn from the same pool and share
its idiosyncrasies (see the wiki note on rbf-kernel-memorization: it's
bounded to the convex hull of what it has seen and fails silently outside
it). This instead re-splits the data several times and picks the param
combo whose discriminator AUC stays close to 0.5 *and* stable across
splits -- low mean distance from 0.5, low variance.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import ParameterGrid, train_test_split

from .evaluate import discriminator_auc
from .synthesizer import NoGANSynthesizer


def robustness_search(
    df: pd.DataFrame,
    param_grid: dict,
    n_splits: int = 3,
    sample_size: int = 20_000,
    train_size: float = 0.5,
) -> pd.DataFrame:
    rows = []
    for split_i in range(n_splits):
        sub = df.sample(n=min(sample_size, len(df)), random_state=split_i)
        train, _holdout = train_test_split(
            sub, train_size=train_size, random_state=split_i
        )

        for params in ParameterGrid(param_grid):
            synth = NoGANSynthesizer(random_state=split_i, **params)
            synth.fit(train)
            synthetic = synth.sample(len(train))
            auc = discriminator_auc(train, synthetic)
            rows.append({**params, "split": split_i, "auc": auc})

    results = pd.DataFrame(rows)
    summary = (
        results.groupby([c for c in results.columns if c not in ("split", "auc")])
        .agg(mean_auc=("auc", "mean"), std_auc=("auc", "std"))
        .reset_index()
    )
    summary["dist_from_ideal"] = (summary["mean_auc"] - 0.5).abs()
    return summary.sort_values(["dist_from_ideal", "std_auc"]).reset_index(drop=True)
