# synthetic

Exploratory synthetic tabular data generation for `csv/flat-training.csv`
(80 mixed categorical/numeric columns, 100,000 real rows). The project
compares several synthesis approaches — kernel resampling, tree-structured
conditioning, and autoregressive CART synthesis — scored against real data
with [`mostlyai-qa`](https://github.com/mostly-ai/mostlyai-qa)'s accuracy /
similarity / privacy report.

**Current best method: `AutoregressiveSynthesizer`** — see
[`CLAUDE.md`](CLAUDE.md) for the full comparison and why the other variants
were dropped or kept as baselines.

## Install

Dependency management is via [Poetry](https://python-poetry.org/), Python
`>=3.11,<3.13`.

```bash
poetry install
poetry run pytest tests/                        # run the test suite
poetry run python scripts/run_autoreg_prototype.py   # full QA report, best method
```

`poetry install` also installs `nogan_synth` itself in editable mode (see
`packages` in `pyproject.toml`), so `from nogan_synth import ...` works
anywhere in the repo without a separate `pip install -e`.

There is also an older, notebook-only pipeline (`model_pipeline_draft.ipynb`,
`data_exploration.ipynb`) predating the `nogan_synth` package — run via
`poetry run jupyter notebook`. It depends on `tensorflow`/`keras` (pinned to
older versions for the Keras `Sequential`/SavedModel API); the `nogan_synth`
package does not need them.

## Package: `src/nogan_synth/`

Sklearn-style (`fit`/`sample`) synthesizers plus supporting diagnostic tools.

| Module | What it does |
|---|---|
| `autoreg_synth.py` | **`AutoregressiveSynthesizer`** — the recommended method. Sequential CART synthesis: columns are ordered by association strength, then each column is predicted from every already-generated column via a decision tree; a real training value is sampled from whichever leaf a synthetic row lands in. |
| `synthesizer.py` | `NoGANSynthesizer` — kernel-weighted nearest-neighbor resampling with mixup-style blending. The previous best method; still a useful baseline. Includes `no_blend` (skip blending for specific wide-range columns) and `match_marginals` (rank-based marginal correction — measurably hurts joint structure, kept as a documented negative result). |
| `tree_synth.py` | `TreeKernelSynthesizer` + `association_matrix` — Chow-Liu maximum-spanning-tree conditioning. Loses badly on this dataset (its correlation structure is one dense blob, not tree-shaped) but kept as a tested/documented result. |
| `block_synth.py` | `BlockKernelSynthesizer` — hybrid: samples the dense correlated "block" as one unit via `NoGANSynthesizer`, tree-conditions everything else around it. Better than the pure tree but still loses to mixup/autoregressive. |
| `embeddings.py` | Pluggable row embeddings (`OneHotEmbedding`, `LabelEmbedding`, `UMAPEmbedding`, `Whitened`) used by `NoGANSynthesizer`'s nearest-neighbor search. |
| `evaluate.py` | `run_qa_report` (wraps `mostlyai.qa.report`), `discriminator_auc` (cheap real-vs-synthetic classifier AUC proxy for tuning), `per_column_discriminator_importance` (which columns a discriminator actually uses to tell real from fake). |
| `relationships.py` | `find_sum_relationships`, `correlation_clusters` — detect columns related by a sum, and group columns into correlation clusters. |
| `reweighting.py` | Kernel Mean Matching (`kernel_mean_match`, `joint_kmm_weights`) and its Nyström-approximated large-scale version, for reweighting a synthetic sample toward matching a real joint distribution. |
| `search.py` | `robustness_search` — multi-split hyperparameter search using `discriminator_auc` as the objective. |

## Scripts: `scripts/`

| Script | What it does |
|---|---|
| `run_autoreg_prototype.py` | Full QA report for `AutoregressiveSynthesizer` (the default/best method). |
| `run_nogan_prototype.py` | Full QA report for `NoGANSynthesizer` (mixup baseline). |
| `run_old_method_report.py` | QA report for the old notebook pipeline's saved output, for comparison. |
| `tune_nogan.py` | Hyperparameter sweep for `NoGANSynthesizer` via `search.robustness_search`. |

## Tests

`poetry run pytest tests/` — one test file per package module, covering
correctness (shapes, dtypes, no exact-duplicate leakage where expected) and
the specific findings that drove each design decision (e.g. that
`no_blend` fixes variance shrinkage without hurting joint structure, that
the autoregressive synthesizer reproduces a trivariate interaction the
kernel methods couldn't).

## Data

- `csv/flat-training.csv`, `csv/sequential-training.csv` — real training
  data (not synthetic).
- Generated run outputs (synthetic CSVs, `*-report.html` QA reports,
  `model/`) are left untracked in git — see `CLAUDE.md` for details.
