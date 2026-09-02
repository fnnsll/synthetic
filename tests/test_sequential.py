import numpy as np
import pandas as pd

from nogan_synth import SequentialAutoregressiveSynthesizer, SequentialNoGANSynthesizer


def _toy_sequential(n_groups=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        T = rng.integers(2, 6)
        state = rng.choice(["a", "b", "c"])
        val = rng.normal()
        for t in range(T):
            state = state if rng.random() < 0.8 else rng.choice(["a", "b", "c"])
            val = 0.9 * val + rng.normal(scale=0.2)
            rows.append({"group_id": f"g{g}", "cat": state, "num": val})
    return pd.DataFrame(rows)


def test_sequential_autoreg_shapes_and_group_structure():
    df = _toy_sequential()
    synth = SequentialAutoregressiveSynthesizer(min_samples_leaf=5, random_state=0)
    synth.fit(df)
    out = synth.sample(150)

    assert list(out.columns) == ["group_id", "cat", "num"]
    assert out["group_id"].nunique() == 150
    sizes = out.groupby("group_id").size()
    assert sizes.min() >= 1
    # bootstrap-sampled from the real length distribution, so shouldn't
    # exceed the real max or collapse to all-length-1
    assert sizes.max() <= df.groupby("group_id").size().max()
    assert (sizes > 1).any()
    assert set(out["cat"].unique()) <= set(df["cat"].unique())


def test_sequential_autoreg_preserves_some_autocorrelation():
    # num is an AR(1)-like process (0.9 * prev + noise); synthetic sequences
    # should keep noticeable lag-1 autocorrelation, not regress to iid noise.
    df = _toy_sequential(n_groups=500, seed=1)
    synth = SequentialAutoregressiveSynthesizer(min_samples_leaf=5, random_state=1)
    synth.fit(df)
    out = synth.sample(500)

    def lag1_corr(frame):
        prev = frame.groupby("group_id")["num"].shift(1)
        mask = prev.notna()
        return np.corrcoef(frame["num"][mask], prev[mask])[0, 1]

    assert lag1_corr(out) > 0.5
    assert lag1_corr(df) > 0.5


def test_sequential_nogan_shapes_and_group_structure():
    df = _toy_sequential(n_groups=200, seed=2)
    synth = SequentialNoGANSynthesizer(jitter=0.1, n_neighbors=20, random_state=0)
    synth.fit(df)
    out = synth.sample(120)

    assert list(out.columns) == ["group_id", "cat", "num"]
    assert out["group_id"].nunique() == 120
    sizes = out.groupby("group_id").size()
    assert sizes.min() >= 1
    assert (sizes > 1).any()
    assert set(out["cat"].unique()) <= set(df["cat"].unique())


if __name__ == "__main__":
    test_sequential_autoreg_shapes_and_group_structure()
    test_sequential_autoreg_preserves_some_autocorrelation()
    test_sequential_nogan_shapes_and_group_structure()
    print("ok")
