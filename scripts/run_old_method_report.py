"""Baseline comparison point: evaluate the old notebook pipeline's saved
output (submission_5.csv) with the same mostlyai.qa.report call used for
the NoGAN synthesizer, so the two methods are directly comparable.

The original X_train/X_test split from that run wasn't saved, so this
re-splits csv/flat-training.csv the same way (80/20, unseeded, matching
model_pipeline_draft.ipynb cell 0) as a stand-in trn/hol reference.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from nogan_synth.evaluate import run_qa_report


def main():
    df = pd.read_csv("csv/flat-training.csv")
    X_train, X_test = train_test_split(df, train_size=0.8)

    synthetic = pd.read_csv("submission_5.csv")
    synthetic = synthetic.drop(columns=["Unnamed: 0", "cluster"])

    report_path, metrics = run_qa_report(
        synthetic, X_train, X_test, report_path="old-method-report.html"
    )
    print(f"report: {report_path}")
    print(metrics.model_dump_json(indent=4))


if __name__ == "__main__":
    main()
