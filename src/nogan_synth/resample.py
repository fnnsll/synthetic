"""Model-agnostic post-processing: take any synthetic *pool* and a real
training set, return a better subset. Independent of how the pool was
generated (autoreg, kernel mixup, a GAN, ...).

``select_subset`` -- pick the subset whose binned univariate/bivariate/
trivariate histograms are closest (normalized L1) to the training set's.
This is the metric ``mostlyai.qa`` scores as ``accuracy_overall``, so
minimizing it directly raises that score.

``separation_weight`` (0..1, default 0) blends in a privacy term: each pool
row's nearest-neighbor distance to a real training row (in ``embedding``
space), normalized to [0, 1] across the pool. The swap search then also
prefers to drop low-separation (near-duplicate-risk) rows in favor of
high-separation ones. This term is separable across rows (no interaction
between which rows are already chosen), so ``separation_weight=1.0`` (no
distributional term at all) just converges to "top target_size rows by
distance to nearest real row" -- useful as a standalone experiment to see
how much distributional accuracy that alone costs.

ATTRIBUTION
-----------
The greedy simulated-annealing swap search here is a re-implementation of
the post-processing stage of the winning submission to the MOSTLY AI Prize
Flat Data Challenge (2025):

    author:  Gandagorn  (https://github.com/Gandagorn)
    source:  https://github.com/Gandagorn/mostlyai_flat  (MIT License)
    file:    pipeline/postprocessing.py

Their pipeline is: MOSTLY AI generative model -> oversized synthetic pool
-> IPF selection -> greedy trimming -> iterative annealed refinement. We
rebuilt only the refinement loop (``_make_spec`` / ``_bin_df`` /
``_top_triples`` / ``choose_rows_by_refinement`` -> the code below), to
study why it works, and made it generator-agnostic. Not yet ported: the
IPF warm-start and the separate trimming phase. The re-warming
(``rewarm_patience``) and best-seen-returned logic are additions of ours,
not in the original.
"""
from __future__ import annotations

import time
import warnings
from itertools import combinations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# binning                                                                     #
# --------------------------------------------------------------------------- #


def _make_spec(df: pd.DataFrame, bins: int) -> dict:
    spec = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            edges = np.unique(
                np.quantile(s.dropna().astype("float64"), np.linspace(0, 1, bins + 1))
            )
            if len(edges) < 2:
                edges = np.linspace(np.nan_to_num(s.min()), np.nan_to_num(s.max()) + 1, 2)
            spec[col] = ("num", edges)
        else:
            top = s.value_counts(dropna=False).index[: bins - 1]
            spec[col] = ("cat", {v: i for i, v in enumerate(top)}, bins - 1)
    return spec


def _bin_df(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col, info in spec.items():
        if info[0] == "num":
            edges = info[1]
            out[col] = np.searchsorted(edges[1:-1], df[col].values, side="right")
        else:
            mapping, other = info[1], info[2]
            out[col] = df[col].map(mapping).fillna(other)
    return out.astype("int64")


def _top_pairs(tr_bin: pd.DataFrame, k: int) -> list[tuple[str, str]]:
    from sklearn.metrics import mutual_info_score

    scored = sorted(
        (
            (mutual_info_score(tr_bin[a], tr_bin[b]), (a, b))
            for a, b in combinations(tr_bin.columns, 2)
        ),
        key=lambda t: -t[0],
    )
    return [p for _, p in scored[:k]]


def _top_triples(tr_bin: pd.DataFrame, k: int, bins: int) -> list[tuple[str, str, str]]:
    from sklearn.metrics import mutual_info_score

    if k == 0 or tr_bin.shape[1] < 3:
        return []
    pair_mi = sorted(
        (
            (mutual_info_score(tr_bin[a], tr_bin[b]), (a, b))
            for a, b in combinations(tr_bin.columns, 2)
        ),
        key=lambda t: -t[0],
    )
    pool_cols = {c for _, pr in pair_mi[: max(1, 4 * k)] for c in pr}
    if len(pool_cols) < 3:
        return []
    scored = []
    for a, b, c in combinations(sorted(pool_cols), 3):
        joint = tr_bin[a].values * bins + tr_bin[b].values
        scored.append((mutual_info_score(joint, tr_bin[c].values), (a, b, c)))
    scored.sort(key=lambda t: -t[0])
    return [t for _, t in scored[:k]]


# --------------------------------------------------------------------------- #
# subset selection                                                            #
# --------------------------------------------------------------------------- #


def _composite(bin_np: dict, name_parts: tuple, bins: int) -> np.ndarray:
    vals = bin_np[name_parts[0]].astype(np.int64)
    for p in name_parts[1:]:
        vals = vals * bins + bin_np[p]
    return vals.astype(np.int64)


def _znorm(x: np.ndarray) -> np.ndarray:
    std = x.std()
    return (x - x.mean()) / std if std > 1e-12 else np.zeros_like(x)


def _nearest_real_dist(train_df: pd.DataFrame, pool_df: pd.DataFrame, embedding) -> np.ndarray:
    """Each pool row's distance to its nearest real training row, in
    ``embedding`` space. Normalized to [0, 1] across the pool.
    """
    from sklearn.neighbors import NearestNeighbors

    from .embeddings import resolve_embedding

    emb = resolve_embedding(embedding)
    emb.fit(train_df)
    train_z = emb.transform(train_df)
    pool_z = emb.transform(pool_df)
    dist, _ = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(train_z).kneighbors(pool_z)
    dist = dist[:, 0]
    span = dist.max() - dist.min()
    return (dist - dist.min()) / span if span > 1e-12 else np.zeros_like(dist)


def select_subset(
    train_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    target_size: int | None = None,
    *,
    bins: int = 10,
    top_pairs: int = 40,
    top_triples: int = 5,
    iterations: int = 500,
    swap_size: int = 100,
    candidate_multiplier: int = 3,
    rewarm_patience: int | None = None,
    rewarm_temp: float = 1e-4,
    max_time: float | None = None,
    separation_weight: float = 0.0,
    separation_embedding="onehot",
    random_state: int | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Select ``target_size`` rows from ``pool_df`` matching ``train_df``'s
    binned uni/bi/trivariate distributions (normalized-L1 minimization via
    annealed greedy swapping). Returns the selected rows, dtypes coerced to
    ``train_df``'s.

    The swap size shrinks on rejection, so from a random start the search
    plateaus once improvements get rare (swap size -> 1, and only
    ``candidate_multiplier`` pool rows seen per iteration, so a large pool
    goes unused). Set ``rewarm_patience`` to the number of consecutive
    non-improving iterations after which the swap size is reset to
    ``swap_size`` and the acceptance temperature jumps to ``rewarm_temp`` --
    basin-hopping out of the plateau. The best subset seen across the whole
    run is always what's returned, so re-warming can wander uphill safely.
    """
    rng = np.random.default_rng(random_state)
    target_size = target_size or len(train_df)
    cols = list(train_df.columns)
    G = len(pool_df)
    if target_size >= G:
        warnings.warn(f"target_size {target_size} >= pool size {G}; returning whole pool")
        return pool_df.reset_index(drop=True)

    sep_norm = (
        _nearest_real_dist(train_df[cols], pool_df[cols], separation_embedding)
        if separation_weight > 0
        else None
    )

    def total_cost(l1_err: float, sep_sum: float) -> float:
        if sep_norm is None:
            return l1_err
        cost_sep = 1 - sep_sum / target_size
        return (1 - separation_weight) * l1_err + separation_weight * cost_sep

    spec = _make_spec(train_df[cols], bins)
    tr_bin = _bin_df(train_df[cols], spec)
    pl_bin = _bin_df(pool_df[cols], spec)

    pair_feats = _top_pairs(tr_bin, top_pairs)
    triple_feats = _top_triples(tr_bin, top_triples, bins)

    tr_np = {c: tr_bin[c].values for c in cols}
    pl_np = {c: pl_bin[c].values for c in cols}
    phases = {"uni": [(c,) for c in cols], "bi": pair_feats, "tri": triple_feats}
    phase_cols = {ph: [] for ph in phases}
    for ph, feats in phases.items():
        for parts in feats:
            name = "×".join(parts)
            phase_cols[ph].append(name)
            tr_np[name] = _composite(tr_np, parts, bins)
            pl_np[name] = _composite(pl_np, parts, bins)

    targets, nbins = {}, {}
    for ph, names in phase_cols.items():
        if not names:
            continue
        n = bins ** len(phases[ph][0])
        nbins[ph] = n
        targets[ph] = [np.bincount(tr_np[c], minlength=n) for c in names]

    def l1(hists) -> float:
        total, nph = 0.0, 0
        for ph, tgt in targets.items():
            nph += 1
            err = sum(np.abs(hists[ph][j] - tgt[j]).sum() for j in range(len(tgt)))
            total += err / (2 * len(train_df) * len(tgt))
        return total / nph if nph else 0.0

    chosen = np.zeros(G, dtype=bool)
    chosen[rng.choice(G, size=target_size, replace=False)] = True

    hists = {
        ph: [np.bincount(pl_np[c][chosen], minlength=nbins[ph]) for c in names]
        for ph, names in phase_cols.items()
        if names
    }
    l1_err = l1(hists)
    sep_sum = sep_norm[chosen].sum() if sep_norm is not None else 0.0
    err = total_cost(l1_err, sep_sum)
    if verbose:
        msg = f"initial normalized L1: {l1_err:.6f}"
        if sep_norm is not None:
            msg += f", mean separation: {sep_sum / target_size:.4f}, cost: {err:.6f}"
        print(msg)

    start = time.time()
    cur_swap, temp0 = swap_size, 1e-5
    best_chosen, best_err = chosen.copy(), err
    stall, last_rewarm = 0, -(10**9)
    for i in range(iterations):
        if max_time and (time.time() - start) / 60 > max_time:
            break
        idx_chosen = np.where(chosen)[0]
        idx_pool = np.where(~chosen)[0]
        if len(idx_pool) < cur_swap:
            break

        if rewarm_patience and stall >= rewarm_patience:
            cur_swap, stall, last_rewarm = swap_size, 0, i
            if verbose:
                print(f"iter {i + 1}: re-warm (swap -> {swap_size}), best L1 {best_err:.6f}")

        temp = temp0 * (1 - i / iterations) ** 2
        if rewarm_patience and i - last_rewarm < rewarm_patience:
            temp = max(temp, rewarm_temp)

        # rows currently in whose removal most reduces L1
        rem_gain = np.zeros(len(idx_chosen))
        for ph, tgt in targets.items():
            for j, c in enumerate(phase_cols[ph]):
                denom = tgt[j].sum() * 2
                if denom == 0:
                    continue
                resid = tgt[j] - hists[ph][j]
                v = pl_np[c][idx_chosen]
                rem_gain += (np.abs(resid[v]) - np.abs(resid[v] + 1)) / denom
        if sep_norm is not None:
            # rows close to a real training row (low separation) are good to drop
            rem_gain = (1 - separation_weight) * _znorm(rem_gain) + separation_weight * _znorm(
                -sep_norm[idx_chosen]
            )
        worst = idx_chosen[np.argsort(rem_gain)[-cur_swap:]]
        h_worst = {
            ph: [np.bincount(pl_np[c][worst], minlength=nbins[ph]) for c in phase_cols[ph]]
            for ph in targets
        }

        cand = rng.choice(
            idx_pool, size=min(len(idx_pool), cur_swap * candidate_multiplier), replace=False
        )
        add_gain = np.zeros(len(cand))
        for ph, tgt in targets.items():
            for j, c in enumerate(phase_cols[ph]):
                denom = tgt[j].sum() * 2
                if denom == 0:
                    continue
                resid = tgt[j] - (hists[ph][j] - h_worst[ph][j])
                v = pl_np[c][cand]
                add_gain += (np.abs(resid[v]) - np.abs(resid[v] - 1)) / denom
        if sep_norm is not None:
            # rows far from any real training row (high separation) are good to add
            add_gain = (1 - separation_weight) * _znorm(add_gain) + separation_weight * _znorm(
                sep_norm[cand]
            )
        best = cand[np.argsort(add_gain)[-cur_swap:]]
        h_best = {
            ph: [np.bincount(pl_np[c][best], minlength=nbins[ph]) for c in phase_cols[ph]]
            for ph in targets
        }

        new_hists = {
            ph: [hists[ph][j] - h_worst[ph][j] + h_best[ph][j] for j in range(len(targets[ph]))]
            for ph in targets
        }
        new_l1_err = l1(new_hists)
        new_sep_sum = (
            sep_sum - sep_norm[worst].sum() + sep_norm[best].sum() if sep_norm is not None else 0.0
        )
        new_err = total_cost(new_l1_err, new_sep_sum)

        improved = new_err < err
        if improved or (temp > 1e-12 and np.exp((err - new_err) / temp) > rng.random()):
            chosen[worst] = False
            chosen[best] = True
            hists, err, l1_err, sep_sum = new_hists, new_err, new_l1_err, new_sep_sum
            cur_swap = min(swap_size * 2, cur_swap + 1)
        else:
            cur_swap = max(1, cur_swap - 5)
        stall = 0 if improved else stall + 1
        if err < best_err:
            best_chosen, best_err = chosen.copy(), err
        if verbose and (i + 1) % 100 == 0:
            msg = f"iter {i + 1}: swap {cur_swap}, L1 {l1_err:.6f}"
            if sep_norm is not None:
                msg += f", mean separation {sep_sum / target_size:.4f}"
            msg += f", cost {err:.6f} (best {best_err:.6f})"
            print(msg)

    out = pool_df.iloc[np.where(best_chosen)[0]].reset_index(drop=True)
    for c in cols:
        out[c] = out[c].astype(train_df[c].dtype)
    return out


def _group_summary(df: pd.DataFrame, group_col: str, cols: list[str]) -> pd.DataFrame:
    """One row per group: mean (numeric) / mode (categorical) per column,
    sequence length, plus each column's within-group distinct-value count.

    The nunique columns target mostlyai.qa's "distinct categories per
    sequence" coherence component directly -- mean/mode alone describe a
    group's central tendency but say nothing about how much a column
    *varies* within the sequence (a group where a column is constant vs.
    one where it churns every row look identical to mean/mode, but very
    different to that metric). Not covered here: the "sequences per
    distinct category" component (a property of a category value across
    the whole selected set, not summarizable per-group) and "next-column"
    transition coherence (that's the transition *model's* job -- lag-1
    conditioning already targets it directly -- not post-selection's).
    """
    g = df.groupby(group_col, sort=False)
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in cols if c not in num_cols]
    out = g[num_cols].mean() if num_cols else pd.DataFrame(index=g.size().index)
    for c in cat_cols:
        out[c] = g[c].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
    out["_seq_len"] = g.size()
    nunique = g[cols].nunique().add_prefix("_nunique_")
    out = out.join(nunique)
    return out.reset_index()


def drop_exact_duplicate_groups(
    pool_df: pd.DataFrame, train_df: pd.DataFrame, group_col: str
) -> pd.DataFrame:
    """Drop every group in ``pool_df`` that contains at least one row exactly
    matching a real training row. A synthetic row can't be individually
    dropped without leaving a ragged sequence, so this trades pool size for
    cleanliness -- oversample (``pool_multiplier``) to compensate before
    calling this, then ``select_subset_sequential`` down to the target size.
    """
    pool_df = pool_df.reset_index(drop=True)
    cols = [c for c in train_df.columns if c != group_col]
    dup_flag = pool_df[cols].merge(
        train_df[cols].drop_duplicates(), how="left", indicator=True
    )["_merge"].to_numpy() == "both"
    dirty_groups = set(pool_df.loc[dup_flag, group_col])
    return pool_df[~pool_df[group_col].isin(dirty_groups)].reset_index(drop=True)


def select_subset_sequential(
    train_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    group_col: str,
    target_groups: int | None = None,
    **kwargs,
) -> pd.DataFrame:
    """``select_subset`` for grouped/sequential data: whole groups can't be
    partially selected (that would leave ragged sequences), so this
    summarizes each group into one row -- mean of numeric columns, mode of
    categorical columns, plus its sequence length -- and runs ``select_subset``
    on those summaries to pick which *groups* to keep. Returns the full row
    sequences (from ``pool_df``) for the selected groups, group order
    preserved as in ``pool_df``.

    Same idea as the flat-data post-processing (MOSTLY AI Prize winner's
    method, see module docstring), applied at the context/subject level
    since that's the granularity 1:N sequential data can actually be
    resampled at.
    """
    cols = [c for c in train_df.columns if c != group_col]
    train_summary = _group_summary(train_df, group_col, cols)
    pool_summary = _group_summary(pool_df, group_col, cols)
    target_groups = target_groups or len(train_summary)

    picked = select_subset(
        train_summary.drop(columns=group_col), pool_summary, target_groups, **kwargs
    )
    picked_ids = set(picked[group_col])
    out = pool_df[pool_df[group_col].isin(picked_ids)].reset_index(drop=True)
    return out
