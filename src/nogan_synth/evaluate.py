"""Evaluation helpers: the full mostlyai.qa.report, and a cheap discriminator-AUC
proxy for it used by the tuning search (running the full report per candidate
param set would be far too slow).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from .embeddings import LabelEmbedding


def run_qa_report(
    syn: pd.DataFrame,
    trn: pd.DataFrame,
    hol: pd.DataFrame,
    report_path: str | Path = "nogan-report.html",
    random_state: int | None = None,
):
    from mostlyai import qa

    qa.init_logging()
    if random_state is not None:
        qa.set_random_state(random_state)  # matches the prize's per-run seeding
    return qa.report(
        syn_tgt_data=syn, trn_tgt_data=trn, hol_tgt_data=hol, report_path=report_path
    )


def discriminator_auc(real: pd.DataFrame, synthetic: pd.DataFrame, cv: int = 5) -> float:
    """Cross-validated AUC of a classifier trained to tell real from synthetic rows.

    Mirrors mostlyai's discriminator_auc_training_synthetic metric: ~0.5 means
    indistinguishable (good), ~1.0 means trivially separable (memorized/leaked
    structure, bad). Cheap enough to call per candidate in a param search.
    """
    embedding = LabelEmbedding().fit(pd.concat([real, synthetic], ignore_index=True))
    Z_real = embedding.transform(real)
    Z_syn = embedding.transform(synthetic)

    X = np.vstack([Z_real, Z_syn])
    y = np.concatenate([np.zeros(len(Z_real)), np.ones(len(Z_syn))])

    clf = HistGradientBoostingClassifier(max_iter=100)
    proba = cross_val_predict(
        clf, X, y, cv=StratifiedKFold(cv, shuffle=True, random_state=0), method="predict_proba"
    )[:, 1]
    return roc_auc_score(y, proba)


def per_column_discriminator_importance(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    n_repeats: int = 5,
    random_state: int = 0,
) -> pd.DataFrame:
    """Which columns most let a real-vs-synthetic classifier tell them apart.

    Trains the same discriminator as discriminator_auc, one row per
    original column, then permutation-shuffles each column on a held-out
    split and measures the AUC drop -- catches columns that leak
    synthetic-ness jointly with others (a broken triple relationship, a
    categorical column whose distribution mismatches only conditional on a
    numeric one), not just marginal per-column mismatch that a univariate
    check would miss.
    """
    embedding = LabelEmbedding().fit(pd.concat([real, synthetic], ignore_index=True))
    X = np.vstack([embedding.transform(real), embedding.transform(synthetic)])
    y = np.concatenate([np.zeros(len(real)), np.ones(len(synthetic))])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=random_state
    )
    clf = HistGradientBoostingClassifier(max_iter=100, random_state=random_state)
    clf.fit(X_train, y_train)

    result = permutation_importance(
        clf,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=random_state,
    )

    cols = embedding.cat_cols_ + embedding.num_cols_
    dtype = ["categorical"] * len(embedding.cat_cols_) + ["numeric"] * len(embedding.num_cols_)
    return (
        pd.DataFrame(
            {
                "column": cols,
                "dtype": dtype,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
