import numpy as np
import pandas as pd

from nogan_synth.evaluate import per_column_discriminator_importance


def test_flags_the_column_that_actually_differs():
    rng = np.random.default_rng(0)
    n = 1000
    real = pd.DataFrame(
        {
            "giveaway": rng.normal(size=n),
            "noise_a": rng.normal(size=n),
            "cat": rng.choice(["x", "y"], size=n),
        }
    )
    synthetic = real.copy()
    # Shift only one column so it's the only real tell.
    synthetic["giveaway"] = synthetic["giveaway"] + 5.0

    result = per_column_discriminator_importance(real, synthetic, n_repeats=3)
    assert result.iloc[0]["column"] == "giveaway"
    assert set(result["dtype"]) == {"numeric", "categorical"}


if __name__ == "__main__":
    test_flags_the_column_that_actually_differs()
    print("ok")
