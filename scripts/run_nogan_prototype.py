"""Phase-1 test: NoGAN kernel-memorization synthesizer against this repo's data,
evaluated the same way as model_pipeline_draft.ipynb (mostlyai.qa.report on the
same train/holdout split). n_neighbors=10 comes from scripts/tune_nogan.py's
multi-split discriminator-AUC search; jitter=0.0 topped that search but means
every synthetic row is a verbatim training row (100% exact-duplicate rate --
see nogan_synth.reweighting/discussion), so the default here is jitter=0.02
instead, trading a bit of that AUC for actual row-level privacy.
"""
import argparse

from sklearn.model_selection import train_test_split

import pandas as pd

from nogan_synth import NoGANSynthesizer, select_subset
from nogan_synth.evaluate import run_qa_report

N_SAMPLES = 100_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jitter", type=float, default=0.02)
    parser.add_argument("--n-neighbors", type=int, default=10)
    parser.add_argument("--embedding", default="onehot")
    parser.add_argument(
        "--no-blend",
        default="pumpkin,dog,goldfish,mouse",
        help="comma-separated column names to skip continuous blending for -- "
        "these are wide-range but roughly-symmetric numeric columns where mixup's "
        "blend only shrinks variance with no distribution-shape benefit; a single "
        "real weighted-pick value instead measurably improves univariate accuracy "
        "(pumpkin 0.943->0.965) with no discriminator-AUC cost (see nogan-noblend4-report.html)",
    )
    parser.add_argument("--cat-resample", default="copy", choices=["copy", "block", "kernel"],
                        help="categorical draw: copy one neighbor's whole tuple (default), "
                        "or recombine per correlated block / per column to break record copying")
    parser.add_argument("--report-path", default="nogan-report.html")
    parser.add_argument("--out-csv", default="submission_nogan_1.csv")
    parser.add_argument(
        "--prize-holdout",
        nargs="?",
        const="csv/prize/stage1/flat-holdout.csv.gz",
        help="score against the real MOSTLY AI Prize holdout (run "
        "scripts/ingest_prize_eval.py first) instead of a random 80/20 split; "
        "fits on the full training set. Optional path overrides the default.",
    )
    parser.add_argument("--pool-multiplier", type=float, default=1.0,
                        help="sample POOL_MULTIPLIER*100k rows, then select 100k from them")
    parser.add_argument("--select", action="store_true",
                        help="run marginal-matching subset selection on the pool")
    parser.add_argument("--select-iterations", type=int, default=500)
    parser.add_argument("--select-rewarm", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv("csv/flat-training.csv")
    if args.prize_holdout:
        X_train, X_test = df, pd.read_csv(args.prize_holdout)
    else:
        X_train, X_test = train_test_split(df, train_size=0.8)

    no_blend = [c for c in args.no_blend.split(",") if c]
    synth = NoGANSynthesizer(
        embedding=args.embedding,
        jitter=args.jitter,
        n_neighbors=args.n_neighbors,
        no_blend=no_blend,
        random_state=42,
    )
    synth.fit(X_train)

    synthetic = synth.sample(int(N_SAMPLES * args.pool_multiplier))

    if args.select:
        synthetic = select_subset(X_train, synthetic, N_SAMPLES,
                                  iterations=args.select_iterations,
                                  rewarm_patience=args.select_rewarm,
                                  random_state=42, verbose=True)
    elif len(synthetic) > N_SAMPLES:
        synthetic = synthetic.sample(N_SAMPLES, random_state=42).reset_index(drop=True)

    synthetic.to_csv(args.out_csv, index=False)

    report_path, metrics = run_qa_report(
        synthetic, X_train, X_test, report_path=args.report_path
    )
    print(f"report: {report_path}")
    print(metrics.model_dump_json(indent=4))


if __name__ == "__main__":
    main()
