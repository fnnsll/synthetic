"""Full QA report for AutoregressiveSynthesizer (sequential CART synthesis),
evaluated the same way as run_nogan_prototype.py so the two are directly
comparable. min_samples_leaf=2 is the smallest leaf size that still gets
zero exact-duplicate rows (1 is memorization: ~98% duplicates) and the best
discriminator AUC among non-memorizing settings found in tuning.
"""
import argparse

from sklearn.model_selection import train_test_split

import pandas as pd

from nogan_synth import AutoregressiveSynthesizer
from nogan_synth.evaluate import run_qa_report

N_SAMPLES = 100_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--report-path", default="autoreg-report.html")
    parser.add_argument("--out-csv", default="submission_autoreg_1.csv")
    args = parser.parse_args()

    df = pd.read_csv("csv/flat-training.csv")
    X_train, X_test = train_test_split(df, train_size=0.8)

    synth = AutoregressiveSynthesizer(min_samples_leaf=args.min_samples_leaf, random_state=42)
    synth.fit(X_train)
    synthetic = synth.sample(N_SAMPLES)

    synthetic.to_csv(args.out_csv, index=False)

    report_path, metrics = run_qa_report(
        synthetic, X_train, X_test, report_path=args.report_path
    )
    print(f"report: {report_path}")
    print(metrics.model_dump_json(indent=4))


if __name__ == "__main__":
    main()
