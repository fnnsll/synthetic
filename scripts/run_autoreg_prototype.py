"""Full QA report for AutoregressiveSynthesizer (sequential CART synthesis),
evaluated the same way as run_nogan_prototype.py so the two are directly
comparable. min_samples_leaf=2 is the smallest leaf size that still gets
zero exact-duplicate rows (1 is memorization: ~98% duplicates) and the best
discriminator AUC among non-memorizing settings found in tuning.
"""
import argparse

from sklearn.model_selection import train_test_split

import pandas as pd

from nogan_synth import AutoregressiveSynthesizer, select_subset
from nogan_synth.evaluate import run_qa_report

N_SAMPLES = 100_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--report-path", default="autoreg-report.html")
    parser.add_argument("--out-csv", default="submission_autoreg_1.csv")
    parser.add_argument(
        "--prize-holdout",
        nargs="?",
        const="csv/prize/stage1/flat-holdout.csv.gz",
        help="score against the real MOSTLY AI Prize holdout (run "
        "scripts/ingest_prize_eval.py first) instead of a random 80/20 split; "
        "fits on the full training set. Optional path overrides the default.",
    )
    parser.add_argument(
        "--pool-multiplier",
        type=float,
        default=1.0,
        help="sample POOL_MULTIPLIER*100k rows, then select 100k from them",
    )
    parser.add_argument(
        "--select",
        action="store_true",
        help="run marginal-matching subset selection on the pool",
    )
    parser.add_argument("--select-iterations", type=int, default=500)
    parser.add_argument("--select-rewarm", type=int, default=None,
                        help="select_subset rewarm_patience: basin-hop out of the swap-size "
                        "plateau after this many non-improving iters (try ~iterations/20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="shared random_state for synth fit/sample, select, and qa split")
    args = parser.parse_args()

    df = pd.read_csv("csv/flat-training.csv")
    if args.prize_holdout:
        X_train, X_test = df, pd.read_csv(args.prize_holdout)
    else:
        X_train, X_test = train_test_split(df, train_size=0.8)

    synth = AutoregressiveSynthesizer(
        min_samples_leaf=args.min_samples_leaf, random_state=args.seed
    )
    synth.fit(X_train)

    pool_n = int(N_SAMPLES * args.pool_multiplier)
    synthetic = synth.sample(pool_n)

    if args.select:
        synthetic = select_subset(
            X_train, synthetic, N_SAMPLES,
            iterations=args.select_iterations, rewarm_patience=args.select_rewarm,
            random_state=args.seed, verbose=True,
        )
    elif len(synthetic) > N_SAMPLES:
        synthetic = synthetic.sample(N_SAMPLES, random_state=args.seed).reset_index(drop=True)

    synthetic.to_csv(args.out_csv, index=False)

    report_path, metrics = run_qa_report(
        synthetic, X_train, X_test, report_path=args.report_path,
        random_state=args.seed,
    )
    print(f"report: {report_path}")
    print(metrics.model_dump_json(indent=4))


if __name__ == "__main__":
    main()
