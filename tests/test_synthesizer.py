import numpy as np
import pandas as pd

from nogan_synth import NoGANSynthesizer


def _toy_df(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "cat_a": rng.choice(["red", "green", "blue"], size=n),
            "cat_b": rng.choice(["x", "y"], size=n),
            "num_a": rng.normal(size=n),
            "num_b": rng.integers(0, 100, size=n),
        }
    )


def test_fit_sample_shapes_and_dtypes():
    df = _toy_df()
    synth = NoGANSynthesizer(embedding="onehot", random_state=0)
    synth.fit(df)
    out = synth.sample(200)

    assert out.shape == (200, df.shape[1])
    assert list(out.columns) == list(df.columns)
    assert set(out["cat_a"].unique()) <= set(df["cat_a"].unique())
    assert set(out["cat_b"].unique()) <= set(df["cat_b"].unique())


def test_zero_jitter_reproduces_real_rows():
    df = _toy_df()
    synth = NoGANSynthesizer(embedding="onehot", jitter=0.0, random_state=0)
    synth.fit(df)
    out = synth.sample(50)

    merged = out.merge(df, how="left", indicator=True)
    assert (merged["_merge"] == "both").all()


def test_label_embedding_also_works():
    df = _toy_df()
    synth = NoGANSynthesizer(embedding="label", random_state=0)
    synth.fit(df)
    out = synth.sample(30)
    assert out.shape == (30, df.shape[1])


def _toy_df_with_correlated_missingness(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    group = rng.choice([True, False], size=n, p=[0.4, 0.6])
    num_a = rng.normal(size=n)
    num_b = rng.normal(size=n)
    # a and b are missing together (same group), not independently.
    num_a = np.where(group, num_a, np.nan)
    num_b = np.where(group, num_b, np.nan)
    return pd.DataFrame({"num_a": num_a, "num_b": num_b, "num_c": rng.normal(size=n)})


def test_blending_does_not_poison_valid_values_with_nan():
    # A blend of a real value with a NaN neighbor must not produce NaN
    # more often than either source row would alone -- averaging (1-lam)*x
    # + lam*NaN unconditionally does exactly that.
    df = _toy_df_with_correlated_missingness()
    synth = NoGANSynthesizer(embedding="onehot", jitter=0.3, n_neighbors=30, random_state=0)
    synth.fit(df)
    out = synth.sample(5000)

    real_rate = df["num_a"].isna().mean()
    synthetic_rate = out["num_a"].isna().mean()
    assert synthetic_rate < real_rate + 0.05


def test_correlated_missingness_roughly_preserved():
    df = _toy_df_with_correlated_missingness()
    synth = NoGANSynthesizer(embedding="onehot", jitter=0.1, n_neighbors=30, random_state=0)
    synth.fit(df)
    out = synth.sample(5000)

    real_co = (df["num_a"].isna() == df["num_b"].isna()).mean()
    synthetic_co = (out["num_a"].isna() == out["num_b"].isna()).mean()
    assert synthetic_co > 0.9  # a/b are missing together essentially always in this toy data
    assert abs(synthetic_co - real_co) < 0.1


def _corr_cat_df(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    # block (a,b) always agree; block (c,d) always agree; e is independent
    # high-cardinality noise so neighbor windows are heterogeneous
    ab = rng.choice(["p", "q", "r"], size=n)
    cd = rng.choice(["m", "n"], size=n)
    e = rng.integers(0, 30, size=n).astype(str)
    return pd.DataFrame({"a": ab, "b": ab, "c": cd, "d": cd, "e": e, "num": rng.normal(size=n)})


def test_cat_resample_block_derives_blocks_and_keeps_block_joint():
    df = _corr_cat_df()
    blk = NoGANSynthesizer(
        embedding="onehot", jitter=0.02, n_neighbors=40,
        cat_resample="block", cat_block_threshold=0.15, random_state=0,
    )
    blk.fit(df)
    # correlated pairs land together, independent noise column stays separate
    assert {frozenset(b) for b in blk.cat_blocks_} >= {frozenset({"a", "b"}), frozenset({"c", "d"})}
    assert any(b == ["e"] for b in blk.cat_blocks_)

    out = blk.sample(3000)
    # within-block joint copied intact from real rows
    assert (out["a"] == out["b"]).mean() > 0.99
    assert (out["c"] == out["d"]).mean() > 0.99


def test_cat_resample_kernel_runs():
    df = _corr_cat_df()
    synth = NoGANSynthesizer(
        embedding="onehot", jitter=0.02, n_neighbors=40, cat_resample="kernel", random_state=0
    )
    synth.fit(df)
    out = synth.sample(1000)
    assert out.shape == (1000, df.shape[1])
    assert set(out["e"].unique()) <= set(df["e"].unique())


if __name__ == "__main__":
    test_fit_sample_shapes_and_dtypes()
    test_zero_jitter_reproduces_real_rows()
    test_label_embedding_also_works()
    test_blending_does_not_poison_valid_values_with_nan()
    test_correlated_missingness_roughly_preserved()
    print("ok")
