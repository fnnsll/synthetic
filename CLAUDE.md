# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An exploratory synthetic tabular data generation project. There is no application code (no `.py` source, no package) — the entire pipeline lives in two Jupyter notebooks. All work happens by editing/running notebook cells.

- `data_exploration.ipynb` — early exploration of `csv/flat-training.csv`: correlation structure between columns, cosine-distance/similarity between rows, connected-components analysis via `networkx` to find groups of related columns.
- `model_pipeline_draft.ipynb` — the main pipeline: label-encodes categorical columns, finds correlated column clusters (`networkx` connected components) and separates single/unclustered columns, fits `umap.UMAP` (hamming metric) to embed categorical columns and assigns clusters visually, resamples/augments rows per column-cluster to build a synthetic candidate pool (`random_dataframe_from_col_space.csv`), filters candidates using pairwise-distance ("bottom degrees") comparison against real training rows, re-embeds the filtered candidates and drops UMAP-noise cluster (`-1`), then trains a Keras `Sequential` model (encoder/decoder pattern), and finally scores the synthetic output against real train/holdout data using `mostlyai.qa.report`.
- `model_pipeline_draft copy.ipynb` — a checkpoint/variant of the draft above; treat as historical unless told otherwise.

## Data layout

- `csv/flat-training.csv`, `csv/sequential-training.csv` — real training data (not synthetic).
- `random_dataframe_from_col_space.csv`, `fake_sample.csv`, `submission_*.csv` — generated/intermediate synthetic outputs from notebook runs. These are reproducible artifacts, not hand-maintained data; regenerate by re-running the relevant notebook cells rather than editing directly.
- `model/o_model/` — saved model artifacts: `model.pkl` plus Keras `encoder/`, `decoder/`, `parametric_model/` SavedModel directories, produced by the training cells in `model_pipeline_draft.ipynb`.
- `model-report.html` — the QA report produced by `mostlyai.qa.report(...)` comparing synthetic vs. real train/holdout data.
- `output*.png` — saved plots (UMAP embeddings, outlier plots, cluster scatter) referenced from the notebooks.

## Environment

Dependency management is via Poetry (`pyproject.toml` / `poetry.lock`), Python `>=3.11,<3.13`.

```bash
poetry install       # set up the environment
poetry shell          # or: poetry run jupyter notebook
jupyter notebook      # open model_pipeline_draft.ipynb / data_exploration.ipynb
```

Key libraries in play: `pandas`, `numpy`, `scikit-learn`, `networkx`, `umap-learn`, `tensorflow`/`keras` (<=2.13 / <=3, pinned — do not upgrade casually, notebooks depend on the older Keras `Sequential`/SavedModel API), `seaborn`/`matplotlib` for plotting, `mostlyai-qa` for synthetic-data quality scoring.

`black` and `isort` are configured (line-length 100) but there is no source tree to lint since all code is in notebooks. `pytest` is a listed dependency but there are no test files in the repo.

## Working conventions

- Notebook cells build on prior cell state (e.g. `X_train`, `corr_df`, `component_list`, `updated_components`, `embedding_o` are defined earlier and reused many cells later) — cells are not independently runnable; execute top-to-bottom.
- When editing a notebook, prefer adding/modifying cells over restructuring the whole notebook; the pipeline is a linear exploratory script, not a modular library.
- Large generated CSVs and model artifacts are checked into the working tree as run outputs — don't assume they're meant to be committed to git; check with the user before adding them.
