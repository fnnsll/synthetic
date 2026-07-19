import numpy as np
import pandas as pd

from nogan_synth import joint_kmm_weights, joint_kmm_weights_nystrom, weighted_resample


def _trivariate_df(n, seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    c = a * b + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"a": a, "b": b, "c": c})


def test_kmm_recovers_broken_trivariate_interaction():
    real = _trivariate_df(2000, seed=0)
    # A synthetic pool with each column resampled independently from its own
    # marginal -- correct univariate marginals, correct pairwise correlations
    # (all ~0 here), but the a*b ~ c three-way interaction is destroyed.
    rng = np.random.default_rng(1)
    synthetic = pd.DataFrame(
        {
            "a": rng.choice(real["a"], size=4000, replace=True),
            "b": rng.choice(real["b"], size=4000, replace=True),
            "c": rng.choice(real["c"], size=4000, replace=True),
        }
    )

    true_interaction_corr = np.corrcoef(real["a"] * real["b"], real["c"])[0, 1]
    broken_corr = np.corrcoef(synthetic["a"] * synthetic["b"], synthetic["c"])[0, 1]
    assert broken_corr < true_interaction_corr - 0.3  # confirm it's actually broken

    weights = joint_kmm_weights(real, synthetic, cols=["a", "b", "c"], weight_cap=20.0)
    reweighted = weighted_resample(synthetic, weights, n=4000, random_state=0)
    fixed_corr = np.corrcoef(reweighted["a"] * reweighted["b"], reweighted["c"])[0, 1]

    assert abs(fixed_corr - true_interaction_corr) < abs(broken_corr - true_interaction_corr)


def test_nystrom_recovers_broken_trivariate_interaction():
    real = _trivariate_df(2000, seed=0)
    rng = np.random.default_rng(1)
    synthetic = pd.DataFrame(
        {
            "a": rng.choice(real["a"], size=4000, replace=True),
            "b": rng.choice(real["b"], size=4000, replace=True),
            "c": rng.choice(real["c"], size=4000, replace=True),
        }
    )

    true_interaction_corr = np.corrcoef(real["a"] * real["b"], real["c"])[0, 1]
    broken_corr = np.corrcoef(synthetic["a"] * synthetic["b"], synthetic["c"])[0, 1]

    # Nystrom with all pooled points as landmarks should behave like the
    # exact QP -- same directional fix on the broken interaction.
    weights = joint_kmm_weights_nystrom(
        real, synthetic, cols=["a", "b", "c"], weight_cap=20.0, n_landmarks=400
    )
    reweighted = weighted_resample(synthetic, weights, n=4000, random_state=0)
    fixed_corr = np.corrcoef(reweighted["a"] * reweighted["b"], reweighted["c"])[0, 1]

    assert abs(fixed_corr - true_interaction_corr) < abs(broken_corr - true_interaction_corr)


def test_nystrom_approximates_exact_kmm():
    # Small enough for the exact O(n^2) QP to be a ground truth to compare against.
    real = _trivariate_df(300, seed=0)
    synthetic = _trivariate_df(300, seed=2)

    exact = joint_kmm_weights(real, synthetic, cols=["a", "b", "c"], weight_cap=20.0)
    approx = joint_kmm_weights_nystrom(
        real, synthetic, cols=["a", "b", "c"], weight_cap=20.0, n_landmarks=250
    )
    # Not identical (low-rank approximation), but should correlate strongly
    # with the exact solution rather than being unrelated noise.
    assert np.corrcoef(exact, approx)[0, 1] > 0.8


if __name__ == "__main__":
    test_kmm_recovers_broken_trivariate_interaction()
    test_nystrom_recovers_broken_trivariate_interaction()
    test_nystrom_approximates_exact_kmm()
    print("ok")
