import numpy as np
import pandas as pd

from nogan_synth import TreeKernelSynthesizer, association_matrix


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
    synth = TreeKernelSynthesizer(random_state=0)
    synth.fit(df)
    out = synth.sample(200)

    assert out.shape == (200, df.shape[1])
    assert list(out.columns) == list(df.columns)
    assert set(out["cat_a"].unique()) <= set(df["cat_a"].unique())
    assert set(out["cat_b"].unique()) <= set(df["cat_b"].unique())


def test_zero_jitter_reproduces_real_rows_along_edges():
    df = _toy_df()
    df["num_a"] = df["num_a"].round(3)  # realistic quantized precision, not raw float64
    synth = TreeKernelSynthesizer(jitter=0.0, random_state=0)
    synth.fit(df)
    out = synth.sample(50)
    # Every value must come from *some* real row's value for that column
    # (jitter=0 means each edge picks a real value verbatim, though not
    # necessarily all from the same source row).
    for col in df.columns:
        assert set(out[col].unique()) <= set(df[col].unique())


def _correlated_triple_df(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = a * 0.9 + rng.normal(scale=0.2, size=n)
    c = b * 0.9 + rng.normal(scale=0.2, size=n)
    # An independent noise column so the tree has more than one real edge to pick among.
    d = rng.normal(size=n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d})


def test_association_matrix_ranks_the_true_chain():
    df = _correlated_triple_df()
    assoc = association_matrix(df)
    assert assoc.loc["a", "b"] > assoc.loc["a", "d"]
    assert assoc.loc["b", "c"] > assoc.loc["b", "d"]


def test_tree_reproduces_chain_correlation():
    df = _correlated_triple_df()
    true_corr_ab = df["a"].corr(df["b"])
    true_corr_bc = df["b"].corr(df["c"])
    true_corr_ac = df["a"].corr(df["c"])

    synth = TreeKernelSynthesizer(n_neighbors=30, jitter=0.05, random_state=0)
    synth.fit(df)
    out = synth.sample(2000)

    # b's tree-parent should be a (or vice versa) and c's should be b --
    # the strongest real edges -- so those pairwise correlations should
    # survive reasonably well through the chain.
    assert abs(out["a"].corr(out["b"]) - true_corr_ab) < 0.15
    assert abs(out["b"].corr(out["c"]) - true_corr_bc) < 0.15
    # a/c are only connected through b (tree/Markov assumption), so some
    # falloff vs. the true direct correlation is expected and fine.
    assert out["a"].corr(out["c"]) > 0


def test_mixed_categorical_numeric_association():
    rng = np.random.default_rng(0)
    n = 1000
    cat = rng.choice(["lo", "hi"], size=n)
    num = np.where(cat == "hi", rng.normal(5, 1, n), rng.normal(0, 1, n))
    noise = rng.normal(size=n)
    df = pd.DataFrame({"cat": cat, "num": num, "noise": noise})

    assoc = association_matrix(df)
    assert assoc.loc["cat", "num"] > assoc.loc["cat", "noise"]

    synth = TreeKernelSynthesizer(jitter=0.05, random_state=0)
    synth.fit(df)
    assert ("cat", "num") in synth.bfs_edges_ or ("num", "cat") in synth.bfs_edges_ or (
        synth.root_ in ("cat", "num")
    )


if __name__ == "__main__":
    test_fit_sample_shapes_and_dtypes()
    test_zero_jitter_reproduces_real_rows_along_edges()
    test_association_matrix_ranks_the_true_chain()
    test_tree_reproduces_chain_correlation()
    test_mixed_categorical_numeric_association()
    print("ok")
