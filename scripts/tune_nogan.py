"""Multi-split robustness search over NoGANSynthesizer hyperparameters,
using discriminator AUC as a cheap generalization proxy (see search.py).
Prints the top param combos sorted by closeness to AUC 0.5 and stability
across splits.
"""
import pandas as pd

from nogan_synth.search import robustness_search

PARAM_GRID = {
    "embedding": ["onehot"],
    "jitter": [0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0],
    "n_neighbors": [10, 50, 200],
}


def main():
    df = pd.read_csv("csv/flat-training.csv")
    results = robustness_search(df, PARAM_GRID, n_splits=3, sample_size=20_000)
    pd.set_option("display.width", 120)
    print(results.head(15))
    results.to_csv("nogan_tuning_results.csv", index=False)


if __name__ == "__main__":
    main()
