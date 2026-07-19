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

## nogan_synth package (recommended path)

`src/nogan_synth/` is an installable sklearn-style package (`poetry install` picks it up via `pyproject.toml`'s `packages` entry) that superseded the notebook pipeline's approach. Several synthesizer variants were tried; **`AutoregressiveSynthesizer` (in `autoreg_synth.py`) is the current best/default method** — sequential CART-based conditional synthesis (each column conditioned on every already-generated column via a decision tree, real training value sampled from whichever leaf a synthetic row lands in). At `min_samples_leaf=2` (the class default) it beats every other variant tried on `csv/flat-training.csv`: higher accuracy at every tier (overall 0.979 vs `NoGANSynthesizer` mixup's 0.965, trivariate 0.966 vs 0.951), same discriminator AUC (~0.58), zero exact-duplicate rows. Run `scripts/run_autoreg_prototype.py` for the full QA report.

Other variants, kept as tested/documented negative or partial results (see their module docstrings for why):
- `NoGANSynthesizer` (`synthesizer.py`) — kernel-weighted nearest-neighbor mixup blending. Previously the best method before the autoregressive one; still useful as a comparison baseline. Has a `no_blend` option (default `pumpkin,dog,goldfish,mouse` in the script) for columns where continuous blending only shrinks variance with no benefit.
- `TreeKernelSynthesizer` / `BlockKernelSynthesizer` (`tree_synth.py`, `block_synth.py`) — Chow-Liu tree conditioning; loses badly (AUC ~0.98-0.999) because this dataset's correlation structure is one dense blob, not tree-shaped.
- A Gaussian-copula numeric mode was tried in `NoGANSynthesizer` and removed entirely (see git history / `synthesizer.py` docstring) — broke on point-mass/near-binary numeric columns.
- `reweighting.py` (KMM/Nystrom) and `relationships.py` (sum/correlation-cluster detection) are diagnostic/correction tools, not synthesizers themselves.

`tests/` covers all of the above; run `poetry run pytest tests/`.

## Working conventions

- Notebook cells build on prior cell state (e.g. `X_train`, `corr_df`, `component_list`, `updated_components`, `embedding_o` are defined earlier and reused many cells later) — cells are not independently runnable; execute top-to-bottom.
- When editing a notebook, prefer adding/modifying cells over restructuring the whole notebook; the pipeline is a linear exploratory script, not a modular library.
- Large generated CSVs and model artifacts are checked into the working tree as run outputs — don't assume they're meant to be committed to git; check with the user before adding them.
