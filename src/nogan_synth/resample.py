"""Model-agnostic post-processing: take any synthetic *pool* and a real
training set, return a better subset. Independent of how the pool was
generated (autoreg, kernel mixup, a GAN, ...).

``select_subset`` -- pick the subset whose binned univariate/bivariate/
trivariate histograms are closest (normalized L1) to the training set's.
This is the metric ``mostlyai.qa`` scores as ``accuracy_overall``, so
minimizing it directly raises that score.

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
    err = l1(hists)
    if verbose:
        print(f"initial normalized L1: {err:.6f}")

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
        best = cand[np.argsort(add_gain)[-cur_swap:]]
        h_best = {
            ph: [np.bincount(pl_np[c][best], minlength=nbins[ph]) for c in phase_cols[ph]]
            for ph in targets
        }

        new_hists = {
            ph: [hists[ph][j] - h_worst[ph][j] + h_best[ph][j] for j in range(len(targets[ph]))]
            for ph in targets
        }
        new_err = l1(new_hists)

        improved = new_err < err
        if improved or (temp > 1e-12 and np.exp((err - new_err) / temp) > rng.random()):
            chosen[worst] = False
            chosen[best] = True
            hists, err = new_hists, new_err
            cur_swap = min(swap_size * 2, cur_swap + 1)
        else:
            cur_swap = max(1, cur_swap - 5)
        stall = 0 if improved else stall + 1
        if err < best_err:
            best_chosen, best_err = chosen.copy(), err
        if verbose and (i + 1) % 100 == 0:
            print(f"iter {i + 1}: swap {cur_swap}, L1 {err:.6f} (best {best_err:.6f})")

    out = pool_df.iloc[np.where(best_chosen)[0]].reset_index(drop=True)
    for c in cols:
        out[c] = out[c].astype(train_df[c].dtype)
    return out
