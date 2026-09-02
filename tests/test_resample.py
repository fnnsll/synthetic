import numpy as np
import pandas as pd

from nogan_synth import drop_exact_duplicate_groups, select_subset, select_subset_sequential


def _real(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    return pd.DataFrame(
        {
            "cat": rng.choice(["p", "q", "r", "s"], size=n, p=[0.5, 0.3, 0.15, 0.05]),
            "x": a,
            "y": a + rng.normal(scale=0.3, size=n),
        }
    )


def _biased_pool(n=8000, seed=1):
    # same columns, deliberately wrong marginals (shifted x, flat cat)
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=0.7, size=n)
    return pd.DataFrame(
        {
            "cat": rng.choice(["p", "q", "r", "s"], size=n),
            "x": a,
            "y": a + rng.normal(scale=0.3, size=n),
        }
    )


def _l1(train, syn, bins=8):
    from nogan_synth.resample import _bin_df, _make_spec

    spec = _make_spec(train, bins)
    tb, sb = _bin_df(train, spec), _bin_df(syn, spec)
    err = 0.0
    for c in train.columns:
        t = np.bincount(tb[c], minlength=bins)
        s = np.bincount(sb[c], minlength=bins) * (len(train) / len(syn))
        err += np.abs(t - s).sum() / (2 * len(train))
    return err / train.shape[1]


def test_select_subset_reduces_marginal_error():
    train, pool = _real(), _biased_pool()
    rng = np.random.default_rng(0)
    baseline = pool.iloc[rng.choice(len(pool), len(train), replace=False)]

    picked = select_subset(
        train, pool, len(train), bins=8, iterations=300, swap_size=60, random_state=0
    )

    assert len(picked) == len(train)
    assert list(picked.columns) == list(train.columns)
    assert _l1(train, picked) < _l1(train, baseline) * 0.7


def test_select_subset_returns_whole_pool_when_target_too_large():
    train, pool = _real(500), _real(300, seed=5)
    out = select_subset(train, pool, target_size=400, iterations=10)
    assert len(out) == len(pool)


def test_select_subset_rewarm_runs_and_still_beats_random():
    train, pool = _real(), _biased_pool()
    rng = np.random.default_rng(0)
    baseline = pool.iloc[rng.choice(len(pool), len(train), replace=False)]
    out = select_subset(train, pool, len(train), bins=8, iterations=400,
                        swap_size=60, rewarm_patience=15, random_state=0)
    assert len(out) == len(train)
    assert _l1(train, out) < _l1(train, baseline) * 0.7


def _mean_nn_dist(train, syn):
    from sklearn.neighbors import NearestNeighbors

    from nogan_synth.embeddings import OneHotEmbedding

    emb = OneHotEmbedding().fit(train)
    dist, _ = NearestNeighbors(n_neighbors=1).fit(emb.transform(train)).kneighbors(
        emb.transform(syn)
    )
    return dist.mean()


def test_select_subset_separation_only_maximizes_distance_from_train():
    # pool has some rows that are near-copies of training rows (dup risk) and
    # some far away; separation_weight=1.0 (no distributional term at all)
    # should pick the far ones every time.
    train = _real(500)
    rng = np.random.default_rng(2)
    near = train.sample(1000, replace=True, random_state=2).reset_index(drop=True)
    near["x"] += rng.normal(scale=0.01, size=len(near))
    far = _biased_pool(1000, seed=3)
    far["x"] += 5.0
    pool = pd.concat([near, far], ignore_index=True)

    baseline = pool.sample(500, random_state=0)
    out = select_subset(
        train, pool, 500, separation_weight=1.0, iterations=200, swap_size=100, random_state=0
    )

    assert len(out) == 500
    assert _mean_nn_dist(train, out) > _mean_nn_dist(train, baseline) * 2
    # every selected row should have come from the far half
    assert (out["x"] > 2.0).all()


def test_select_subset_separation_weight_zero_matches_l1_only():
    train, pool = _real(), _biased_pool()
    out_default = select_subset(train, pool, len(train), bins=8, iterations=100,
                                 swap_size=60, random_state=0)
    out_explicit_zero = select_subset(train, pool, len(train), bins=8, iterations=100,
                                       swap_size=60, separation_weight=0.0, random_state=0)
    pd.testing.assert_frame_equal(out_default, out_explicit_zero)


def _grouped(n_groups, seed, loc):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        T = rng.integers(2, 5)
        a = rng.normal(loc=loc)
        for t in range(T):
            rows.append(
                {
                    "group_id": f"g{seed}_{g}",
                    "cat": rng.choice(["p", "q", "r", "s"], p=[0.5, 0.3, 0.15, 0.05]),
                    "x": a + rng.normal(scale=0.2),
                }
            )
    return pd.DataFrame(rows)


def test_select_subset_sequential_keeps_whole_groups_and_improves_fit():
    train = _grouped(300, seed=0, loc=0.0)
    pool = _grouped(1500, seed=1, loc=0.6)  # biased pool, wrong "x" location

    picked = select_subset_sequential(
        train, pool, "group_id", target_groups=300,
        bins=8, iterations=200, swap_size=40, random_state=0,
    )

    # every selected group is intact (no partial sequences)
    real_sizes = dict(zip(pool["group_id"], pool.groupby("group_id")["group_id"].transform("size")))
    for gid, sub in picked.groupby("group_id"):
        assert len(sub) == real_sizes[gid]

    rng = np.random.default_rng(0)
    baseline_ids = rng.choice(pool["group_id"].unique(), size=300, replace=False)
    baseline = pool[pool["group_id"].isin(baseline_ids)]

    assert abs(picked["x"].mean() - train["x"].mean()) < abs(baseline["x"].mean() - train["x"].mean())


def test_drop_exact_duplicate_groups_removes_whole_group_on_any_match():
    train = _grouped(50, seed=0, loc=0.0)
    pool = _grouped(50, seed=0, loc=0.0)  # same seed -> every row matches exactly
    clean = drop_exact_duplicate_groups(pool, train, "group_id")
    assert len(clean) == 0

    pool_mixed = pd.concat([_grouped(50, seed=0, loc=0.0), _grouped(50, seed=1, loc=5.0)])
    clean_mixed = drop_exact_duplicate_groups(pool_mixed, train, "group_id")
    assert clean_mixed["group_id"].nunique() == 50
    assert set(clean_mixed["group_id"]) == set(_grouped(50, seed=1, loc=5.0)["group_id"])


def _grouped_churn(n_groups, seed, churn_p):
    # "cat" flips to a new random value with probability churn_p each step,
    # so within-group nunique(cat) is controlled directly by churn_p while
    # the column's overall marginal distribution stays the same either way.
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        T = rng.integers(4, 8)
        state = rng.choice(["p", "q", "r", "s"])
        for t in range(T):
            if rng.random() < churn_p:
                state = rng.choice(["p", "q", "r", "s"])
            rows.append({"group_id": f"g{seed}_{g}", "cat": state, "x": rng.normal()})
    return pd.DataFrame(rows)


def test_select_subset_sequential_matches_within_group_churn():
    # train has low churn (mostly-constant cat per group); pool is a 50/50
    # mix of low-churn and high-churn groups. Mean/mode alone can't tell
    # these apart (same marginal cat distribution either way) -- only the
    # nunique-per-group summary can steer selection toward the low-churn
    # half, which is what "distinct categories per sequence" coherence
    # measures.
    train = _grouped_churn(300, seed=0, churn_p=0.05)
    pool = pd.concat(
        [_grouped_churn(1000, seed=1, churn_p=0.05), _grouped_churn(1000, seed=2, churn_p=0.9)],
        ignore_index=True,
    )

    picked = select_subset_sequential(
        train, pool, "group_id", target_groups=300,
        bins=8, iterations=300, swap_size=60, random_state=0,
    )

    def mean_nunique(df):
        return df.groupby("group_id")["cat"].nunique().mean()

    rng = np.random.default_rng(0)
    baseline_ids = rng.choice(pool["group_id"].unique(), size=300, replace=False)
    baseline = pool[pool["group_id"].isin(baseline_ids)]

    train_churn = mean_nunique(train)
    picked_churn = mean_nunique(picked)
    baseline_churn = mean_nunique(baseline)

    assert abs(picked_churn - train_churn) < abs(baseline_churn - train_churn)
    # sanity: baseline (50/50 mix) actually sits between the two extremes,
    # so there's real signal here to select away from
    assert baseline_churn > train_churn + 0.3


if __name__ == "__main__":
    test_select_subset_reduces_marginal_error()
    test_select_subset_returns_whole_pool_when_target_too_large()
    test_select_subset_rewarm_runs_and_still_beats_random()
    test_select_subset_separation_only_maximizes_distance_from_train()
    test_select_subset_separation_weight_zero_matches_l1_only()
    test_select_subset_sequential_keeps_whole_groups_and_improves_fit()
    test_drop_exact_duplicate_groups_removes_whole_group_on_any_match()
    test_select_subset_sequential_matches_within_group_churn()
    print("ok")
