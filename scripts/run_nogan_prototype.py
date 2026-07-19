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

from nogan_synth import NoGANSynthesizer
from nogan_synth.evaluate import run_qa_report

N_SAMPLES = 100_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jitter", type=float, default=0.02)
    parser.add_argument("--n-neighbors", type=int, default=10)
    parser.add_argument("--embedding", default="onehot")
    parser.add_argument("--report-path", default="nogan-report.html")
    parser.add_argument("--out-csv", default="submission_nogan_1.csv")
    args = parser.parse_args()

    df = pd.read_csv("csv/flat-training.csv")
    X_train, X_test = train_test_split(df, train_size=0.8)

    synth = NoGANSynthesizer(
        embedding=args.embedding,
        jitter=args.jitter,
        n_neighbors=args.n_neighbors,
        random_state=42,
    )
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
