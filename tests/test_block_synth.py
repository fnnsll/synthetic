import numpy as np
import pandas as pd

from nogan_synth import BlockKernelSynthesizer


def _blocked_df(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    # A dense correlated "block" of 4 columns, all pairwise linked through
    # a shared latent factor (the kind of blob a single tree edge can't
    # represent well) plus two "outside" columns: one tied to the block,
    # one fully independent.
    latent = rng.normal(size=n)
    block_a = latent + rng.normal(scale=0.2, size=n)
    block_b = latent + rng.normal(scale=0.2, size=n)
    block_c = latent + rng.normal(scale=0.2, size=n)
    block_d = latent + rng.normal(scale=0.2, size=n)
    tied_to_block = block_a * 0.8 + rng.normal(scale=0.3, size=n)
    independent = rng.normal(size=n)
    return pd.DataFrame(
        {
            "block_a": block_a,
            "block_b": block_b,
            "block_c": block_c,
            "block_d": block_d,
            "tied_to_block": tied_to_block,
            "independent": independent,
        }
    )


def test_fit_sample_shapes():
    df = _blocked_df()
    block_cols = ["block_a", "block_b", "block_c", "block_d"]
    synth = BlockKernelSynthesizer(block_cols=block_cols, jitter=0.05, random_state=0)
    synth.fit(df)
    out = synth.sample(500)
    assert out.shape == (500, df.shape[1])
    assert list(out.columns) == list(df.columns)


def test_block_internal_correlation_preserved():
    df = _blocked_df()
    block_cols = ["block_a", "block_b", "block_c", "block_d"]
    synth = BlockKernelSynthesizer(
        block_cols=block_cols, block_jitter=0.05, jitter=0.05, random_state=0
    )
    synth.fit(df)
    out = synth.sample(3000)

    true_corr = df["block_a"].corr(df["block_c"])
    syn_corr = out["block_a"].corr(out["block_c"])
    assert abs(syn_corr - true_corr) < 0.1


def test_outside_column_tied_to_block_conditions_on_it():
    df = _blocked_df()
    block_cols = ["block_a", "block_b", "block_c", "block_d"]
    synth = BlockKernelSynthesizer(
        block_cols=block_cols, block_jitter=0.05, jitter=0.05, random_state=0
    )
    synth.fit(df)
    out = synth.sample(3000)

    true_corr = df["block_a"].corr(df["tied_to_block"])
    syn_corr = out["block_a"].corr(out["tied_to_block"])
    assert abs(syn_corr - true_corr) < 0.15


def test_no_outside_columns_degenerates_to_block_only():
    df = _blocked_df()[["block_a", "block_b", "block_c", "block_d"]]
    synth = BlockKernelSynthesizer(block_cols=list(df.columns), random_state=0)
    synth.fit(df)
    out = synth.sample(100)
    assert out.shape == (100, 4)
    assert list(out.columns) == list(df.columns)


if __name__ == "__main__":
    test_fit_sample_shapes()
    test_block_internal_correlation_preserved()
    test_outside_column_tied_to_block_conditions_on_it()
    test_no_outside_columns_degenerates_to_block_only()
    print("ok")
