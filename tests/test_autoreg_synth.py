import numpy as np
import pandas as pd

from nogan_synth import AutoregressiveSynthesizer


def _toy_df(n=200, seed=0):
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
    synth = AutoregressiveSynthesizer(random_state=0)
    synth.fit(df)
    out = synth.sample(200)

    assert out.shape == (200, df.shape[1])
    assert list(out.columns) == list(df.columns)
    assert set(out["cat_a"].unique()) <= set(df["cat_a"].unique())
    assert set(out["cat_b"].unique()) <= set(df["cat_b"].unique())
    # Every numeric value must be a real observed value (no interpolation).
    assert set(out["num_a"].unique()) <= set(df["num_a"].unique())


def _trivariate_interaction_df(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    c = a * b + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"a": a, "b": b, "c": c})


def test_captures_trivariate_interaction():
    # The exact case that broke NoGANSynthesizer's independent-marginal
    # resampling and that KMM only partially fixed (corr 0.01 -> 0.87 vs
    # true 0.995) -- full autoregressive conditioning on a and b together
    # should reproduce c's dependence on their product directly.
    df = _trivariate_interaction_df()
    true_interaction_corr = np.corrcoef(df["a"] * df["b"], df["c"])[0, 1]

    synth = AutoregressiveSynthesizer(order=["a", "b", "c"], min_samples_leaf=10, random_state=0)
    synth.fit(df)
    out = synth.sample(3000)

    syn_interaction_corr = np.corrcoef(out["a"] * out["b"], out["c"])[0, 1]
    assert abs(syn_interaction_corr - true_interaction_corr) < 0.15


def test_handles_missing_values_without_crashing():
    df = _toy_df()
    df.loc[::5, "num_a"] = np.nan
    df.loc[::7, "cat_a"] = None

    synth = AutoregressiveSynthesizer(random_state=0)
    synth.fit(df)
    out = synth.sample(100)

    assert out.shape == (100, df.shape[1])
    # Missingness should show up in the output too (real values, some NaN).
    assert out["num_a"].isna().any()


def test_greedy_order_puts_most_associated_column_first():
    df = _trivariate_interaction_df()
    synth = AutoregressiveSynthesizer(random_state=0)
    synth.fit(df)
    assert synth.order_[0] in df.columns
    assert set(synth.order_) == set(df.columns)


if __name__ == "__main__":
    test_fit_sample_shapes_and_dtypes()
    test_captures_trivariate_interaction()
    test_handles_missing_values_without_crashing()
    test_greedy_order_puts_most_associated_column_first()
    print("ok")
