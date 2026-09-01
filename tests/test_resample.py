import numpy as np
import pandas as pd

from nogan_synth import select_subset


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


if __name__ == "__main__":
    test_select_subset_reduces_marginal_error()
    test_select_subset_returns_whole_pool_when_target_too_large()
    test_select_subset_rewarm_runs_and_still_beats_random()
    print("ok")
