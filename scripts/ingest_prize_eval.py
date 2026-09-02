"""Download the MOSTLY AI Prize data (training + holdout + test) into csv/prize/.

The competition holdout was unseen during Stage 1; it is now public in
github.com/mostly-ai/the-prize-eval. We pull just the CSVs we need over raw
HTTP rather than vendoring their repo. Files land gzipped -- pandas reads
.csv.gz transparently, so no decompression step.

    python scripts/ingest_prize_eval.py            # stage1 flat + sequential
    python scripts/ingest_prize_eval.py --stage 2  # stage2 (remapped columns)

Stage 1 flat-training is byte-identical to csv/flat-training.csv; stage 2 uses
the remapped schema (see mappings.json in their repo) and is not directly
comparable.
"""
import argparse
import urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/mostly-ai/the-prize-eval/main"
OUT = Path("csv/prize")

FILES = ["flat-training.csv.gz", "flat-holdout.csv.gz", "flat-test.csv.gz",
         "sequential-training.csv.gz", "sequential-holdout.csv.gz",
         "sequential-test.csv.gz"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1, choices=(1, 2))
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()

    dest = OUT / f"stage{args.stage}"
    dest.mkdir(parents=True, exist_ok=True)

    for name in FILES:
        kind = name.split("-")[0]  # flat / sequential
        url = f"{RAW}/{kind}/stage{args.stage}/{name}"
        target = dest / name
        if target.exists() and not args.force:
            print(f"skip {target} (exists)")
            continue
        print(f"GET {url}")
        urllib.request.urlretrieve(url, target)
        print(f"  -> {target} ({target.stat().st_size:,} bytes)")

    print(f"\ndone. holdout: {dest / 'flat-holdout.csv.gz'}")


if __name__ == "__main__":
    main()
