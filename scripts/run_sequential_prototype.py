"""Full QA report for the sequential synthesizers against
csv/sequential-training.csv, evaluated the same way as
run_autoreg_prototype.py / run_nogan_prototype.py (train/holdout split,
optional oversized pool + select_subset_sequential post-processing,
mostlyai.qa.report -- now via run_sequential_qa_report's tgt_context_key so
coherence metrics are included, not just accuracy/similarity/distances).
"""
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from nogan_synth import (
    SequentialAutoregressiveSynthesizer,
    SequentialNoGANSynthesizer,
    drop_exact_duplicate_groups,
    select_subset_sequential,
)
from nogan_synth.evaluate import run_sequential_qa_report

GROUP_COL = "group_id"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["autoreg", "nogan"], default="autoreg")

    # autoreg knobs
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--max-depth", type=int, default=None)

    # nogan knobs
    parser.add_argument("--jitter", type=float, default=0.1)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--no-blend", default="", help="comma-separated column names")
    parser.add_argument("--cat-resample", default="copy", choices=["copy", "block", "kernel"])
    parser.add_argument("--cat-swap-frac", type=float, default=1.0)

    # post-processing
    parser.add_argument("--pool-multiplier", type=float, default=1.0,
                        help="sample POOL_MULTIPLIER*n_train_groups groups, then select "
                        "n_train_groups from them")
    parser.add_argument("--select", action="store_true",
                        help="run select_subset_sequential on the pool")
    parser.add_argument("--select-iterations", type=int, default=500)
    parser.add_argument("--select-rewarm", type=int, default=None)
    parser.add_argument("--select-separation-weight", type=float, default=0.0)
    parser.add_argument("--dedup", action="store_true",
                        help="before select_subset_sequential, drop every pool group "
                        "containing a row that exactly matches a real training row "
                        "(nogan's main weak point vs autoreg -- needs a bigger "
                        "--pool-multiplier to compensate for the dropped groups)")

    parser.add_argument("--report-path", default="sequential-report.html")
    parser.add_argument("--out-csv", default="submission_sequential.csv")
    parser.add_argument(
        "--prize-holdout",
        nargs="?",
        const="csv/prize/stage1/sequential-holdout.csv.gz",
        help="score against the real MOSTLY AI Prize sequential holdout (run "
        "scripts/ingest_prize_eval.py first) instead of a random 80/20 split; "
        "fits on the full training set. Optional path overrides the default.",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="shared random_state for synth fit/sample, select, and qa split")
    args = parser.parse_args()

    df = pd.read_csv("csv/sequential-training.csv")
    if args.prize_holdout:
        X_train, X_test = df, pd.read_csv(args.prize_holdout)
    else:
        groups = df[GROUP_COL].unique()
        train_groups, test_groups = train_test_split(
            groups, train_size=0.8, random_state=args.seed
        )
        X_train = df[df[GROUP_COL].isin(train_groups)].reset_index(drop=True)
        X_test = df[df[GROUP_COL].isin(test_groups)].reset_index(drop=True)
    n_train_groups = X_train[GROUP_COL].nunique()

    if args.method == "autoreg":
        synth = SequentialAutoregressiveSynthesizer(
            group_col=GROUP_COL, min_samples_leaf=args.min_samples_leaf,
            max_depth=args.max_depth, random_state=args.seed,
        )
    else:
        no_blend = [c for c in args.no_blend.split(",") if c]
        synth = SequentialNoGANSynthesizer(
            group_col=GROUP_COL, jitter=args.jitter, n_neighbors=args.n_neighbors,
            no_blend=no_blend, cat_resample=args.cat_resample,
            cat_swap_frac=args.cat_swap_frac, random_state=args.seed,
        )
    synth.fit(X_train)

    pool_groups = int(n_train_groups * args.pool_multiplier)
    synthetic = synth.sample(pool_groups)

    if args.dedup:
        before = synthetic[GROUP_COL].nunique()
        synthetic = drop_exact_duplicate_groups(synthetic, X_train, GROUP_COL)
        print(f"dedup: {before} -> {synthetic[GROUP_COL].nunique()} groups")

    if args.select:
        synthetic = select_subset_sequential(
            X_train, synthetic, GROUP_COL, n_train_groups,
            iterations=args.select_iterations, rewarm_patience=args.select_rewarm,
            separation_weight=args.select_separation_weight,
            random_state=args.seed, verbose=True,
        )
    elif synthetic[GROUP_COL].nunique() > n_train_groups:
        keep = pd.Series(synthetic[GROUP_COL].unique()).sample(
            n_train_groups, random_state=args.seed
        )
        synthetic = synthetic[synthetic[GROUP_COL].isin(keep)].reset_index(drop=True)

    synthetic.to_csv(args.out_csv, index=False)

    report_path, metrics = run_sequential_qa_report(
        synthetic, X_train, X_test, group_col=GROUP_COL,
        report_path=args.report_path, random_state=args.seed,
    )
    print(f"report: {report_path}")
    print(metrics.model_dump_json(indent=4))


if __name__ == "__main__":
    main()
