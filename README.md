# Pheno HF

Unsupervised, multimodal phenotyping of patients at elevated short-term risk
of heart failure (HF), from Dutch primary-care (GP) electronic health record
data (AWH-ANHA). This is the analysis code behind the accompanying
(unpublished, working-title) manuscript on data-driven, multimodal HF
phenotyping in primary care: a retrospective cohort of 393,764 patients (117
GP practices, Amsterdam/Haarlem/Almere, 2010–2022) is represented
per-patient from structured GP records (episodes, consultations,
medications) and free text, a Random Forest ranks candidate variables, and a
Gaussian Mixture Model (GMM, via [StepMix](https://github.com/Labo-Lacourse/stepmix))
clusters patients into phenotypes. Phenotypes are then evaluated against the
reference TARGET-HF risk model (structured-data-only logistic model for
incident HF) for added predictive value, and against a structured-data-only
version of the same pipeline to assess the contribution of free text.

This repository contains the analysis code only. **No patient data is
included** — you will need your own GP-EHR extract in the schema described
below (or an adaptation of the loading code for your own data source).

## Pipeline

```
0_pre_process (sensitivity analysis only)
                        derive control flwp-start dates matched to case dates
1_pre_process       tag patients, add episodes+consults to patient-centered dicts
1d1_pre_process      apply cohort/feature filters; add medications+measurements
2_pre_process         merge patient-level dicts from two source databases
2d1_pre_process_text  parse free text per patient; apply cohort masking
A_runner               run Top2Vec text embedding; generate topic-based metrics
3_pre_process          flatten patient representation for all non-text variables
4_pre_process           clean up after 3_pre_process; write a single sparse matrix
4d1_pre_process          scale features (MaxAbsScaler); derive age; drop NaNs
5_dim_reduce               group sparse coded variables (ICPC, ATC codes)
B_runner                    variable selection (Random Forest importance)
GMM_preprocess                scale features (RobustScaler)
C_runner                       fit the Gaussian Mixture Model (StepMix)
analyse_results                  summarize resulting clusters
```

`analyse_results` above refers to the family of `analyse_results_*.py`
scripts, not just `analyse_results.py` — e.g. `analyse_results_AIC_vars.py`
(Table 2 predictive-value comparison), `analyse_results_TARGETHF_score.py`,
`analyse_results_ICPC_codes.py` / `_ATC_codes.py`, `analyse_results_missed_HF.py`
(text- vs. ICPC-detected HF cases), `analyse_results_age_distr.py`,
`analyse_results_var_distributions.py`, `analyse_results_PRAC_ids.py`.

This maps onto the manuscript's Methods as follows:

- **Cohort selection & outcome** (≥1 consultation after 1 Jan 2010, age
  ≥35, HF confirmed by manual GP chart review) — `1_pre_process`,
  `1d1_pre_process`; the 2-year observation window ending 3 months before
  diagnosis (`FOLLOW_UP_PERIOD_DAYS` / `FOLLOW_UP_HFPOS_CENS_WINDOW` in
  `src/constants.py`) is applied throughout pre-processing.
- **Patient representation** — structured variables (age/sex, time-weighted
  ICPC/ATC code occurrences, TARGET-HF predictors) via `3_pre_process`/
  `4_pre_process`/`4d1_pre_process`; ICPC/ATC code grouping (ATC to 3rd
  level, ICPC grouped by clinician-defined domain/severity mapping) via
  `5_dim_reduce`; free-text Top2Vec topic modelling (time-weighted and max
  topic-similarity variables per topic, optimal topic count chosen by a
  penalised silhouette score) via `A_runner`/`textEmbeddingT2V.py`.
- **Variable selection** — `B_runner`/`VariableSelection.py`: Random Forest
  importance (4-fold CV) after dropping one of each pair of variables with
  |r| > 0.95, keeping the top variables per category (ICPC/ATC/text/
  remaining, including TARGET-HF predictors), ~10 per category by default.
- **Phenotype derivation** — `GMM_preprocess`/`C_runner`
  (`GaussMMStepMix.py`, `try_stepmix.py`): GMM fit via StepMix across
  candidate cluster counts (`n_components = list(range(3,30))` in
  `C_runner.py`), selecting the best fit by AIC (`StepMixBICScore.score()`
  returns `-aic` by default, `-bic` if called with `use_bic=True`) and
  hard-assigning each patient to their highest-posterior cluster. No
  variance–covariance structure comparison is implemented (manuscript
  Supplementary Methods SM4 describes comparing structures as part of model
  selection; this code always uses `gaussian_full`/`bernoulli` measurement
  types for continuous/binary variables — see `get_inputs_gmm`).
- **Phenotype characterisation & predictive value vs. TARGET-HF** —
  `analyse_results` (`analyse_results.py`, `analyse_results_util.py`) for
  cluster summaries/labels/case rates, and `analyse_results_AIC_vars.py` for
  the AIC-guided stepwise backward logistic regression (4-fold CV) comparing
  the four predictor setups from Table 2 (TARGET-HF only / cluster
  variables only / TARGET-HF + cluster membership / TARGET-HF + cluster
  membership + cluster variables). Only the third setup
  (`ABS_targetHF_vars_and_cluster_ID`) is currently active in
  `setup_vars` there — the other three are commented out and need
  re-enabling to reproduce the full Table 2 comparison. This script is also
  not currently wired into `ops/runs/analyse_results.sh` (only
  `sandbox_generic.py` runs by default there) — invoke it directly.
- **TARGET-HF reference model** — lives in `src/TargetHF_ty/` (its own
  tagging/cohort/ICPC/text definitions under `targethf/`), kept separate
  from the phenotyping pipeline since it implements the previously
  published reference model (Fam Pract. 2023) rather than this study's own
  method; `try_utils.calc_TARGETHF_scores`/`fetch_diag_TARGETHF` bridge the
  two.

Not reproduced in this repo: the manuscript's Supplementary Methods SM1
describes a semi-automated ICPC grouping pass using GPT-3.5-turbo, run in
parallel with the clinician expert mapping and merged into the candidate
variable set. Only the clinician/expert-knowledge grouping path
(`5_dim_reduce.py`, `DIM_REDUC_TECHNIQUE = 'expert_knowledge'`) is included
here; the LLM-assisted grouping step was exploratory/notebook-based and was
not part of this export (see "What's not in this repo" below).

Known open gap: the manuscript's Sensitivity Analysis restricts to patients
with ≥3 consultations and ≥6 months of observation, matching this code's own
inline comments in `try_utils.py::__init_pats_dict_time_unbiased` — but the
constants that actually apply those restrictions, `MIN_N_CONSULTS` /
`MIN_FLWP_WIN_DAYS` in `src/constants.py`, are both currently set to `1`
(i.e. no restriction). Flagged with a comment at the constants themselves;
left unchanged rather than guessed at, since bumping them changes which
patients enter that analysis.

Also worth noting when cross-checking against the manuscript text: its
Methods state TARGET-HF has "fourteen predictor variables", but its own
Table 2 uses `/16` as the denominator for the TARGET-HF variables column,
and this repo's `targetHF_cols` (`src/namespaces.py`) lists 16 — the code
agrees with the manuscript's table, not its prose; this looks like an
inconsistency in the manuscript draft itself rather than something to fix
here.

`ops/runs/run_experiments.sh` is the main entry point and documents this
chain as a single shell pipeline (`src/*.py`, run in order, each script's
CLI args controlling which cached intermediate files it reads/writes) — most
of its steps are commented out in favour of re-running `analyse_results.py`
against a cached GMM output. `ops/runs/run_sens_analysis_no_text.sh` runs
the same pipeline with text-derived variables excluded from the candidate
set (`use_text_vars=F`); this is the manuscript's "benefit of adding
unstructured text" comparison arm, not its formally distinct Sensitivity
Analysis (patients with ≥6 months observation and ≥3 consultations, controls
matched to cases on observation-period start, stratified by age/sex) — the
latter is driven by `0_pre_process.py` (see "Pipeline" above) and is not
currently wrapped in its own `ops/` script. `ops/runs/analyse_results.sh`
re-runs just the post-hoc analysis scripts against existing GMM output.

## Repository layout

```
src/                    pipeline scripts, shared utilities
src/TargetHF_ty/        the reference TARGET-HF risk model (tagging, cohort/
                         ICPC/text definitions, table distillation) used as
                         the predictor set phenotypes are compared against
ops/                    container build/run + pipeline-runner shell scripts
Dockerfile              build for the environment the pipeline runs in
THIRD_PARTY_LICENSES/   licenses for vendored third-party code (Top2Vec, UMAP)
```

## Setup

Build and run the container:

```sh
docker build -t try_pheno:latest .
DATA_DIR=/path/to/current_extract ./ops/run_container.sh
```

`DATA_DIR` should point at a directory with `parquet/` and `csv/` subfolders
holding the GP-EHR extract (see "Data layout" below). If you also have an
older/previous extract to diff against, set `PREV_DATA_DIR` too. Both are
mounted read-only-in-spirit (the pipeline does not write back to them) at
`/hagdb_grand` and `/hagdb` inside the container — override those mount
points via the `PHENO_HF_DATA_DIR` / `PHENO_HF_PREV_DATA_DIR` environment
variables (see `src/constants.py`) if you'd rather not use those names.

Alternatively, install directly with `pip install -r src/requirements.txt`
and set `PHENO_HF_DATA_DIR` / `PHENO_HF_PREV_DATA_DIR` to your local paths.

Note on Python version: the Dockerfile pins `python:3.11-slim` and
`requirements.txt` pins dependency versions (e.g. `tensorflow==2.15`,
`torch==2.2.0`) consistent with a 3.11 environment, but the manuscript's
Software Availability section states Python v3.14 for the reported run.
Compiled `__pycache__/*.cpython-314.pyc` artifacts in this checkout confirm
these scripts have actually been run under CPython 3.14 at some point too —
so the 3.11-pinned Dockerfile/requirements.txt look like the stale artifact
here, not the manuscript. This repo's pinned dependency versions have not
been verified against a 3.14 interpreter; if you need the environment that
actually produced the published numbers, don't assume the 3.11 container
matches without checking.

Then, from inside the container (or your local checkout):

```sh
cd src
bash ../ops/runs/run_experiments.sh
```

## Data layout

The pipeline expects four record types per source database, keyed by
`(person_id, practice_id, ...)` composite IDs (see
`MAPPING_COMPOSITE_KEYS` in `src/constants.py`):

- **patients** — demographics, registration/deregistration dates
- **episodes** — ICPC-coded problem episodes, with derived risk-factor flags
- **journals** — consultation-level records, structured around the SOEPX
  framework (Subjective/Objective/Evaluation/Plan/additional remarks), each
  with its own optional ICPC code and free-text field (`icpc_s`..`icpc_x`,
  `text_s`..`text_x` in `JOURNAL_COLS`)
- **medications** / **measurements** — ATC-coded prescriptions and lab/vitals

Exact expected columns and dtypes are documented in `CSV_DTYPES` in
`src/constants.py`. Adapting the pipeline to a different EHR export means
matching (or remapping to) this schema in `1_pre_process.py` /
`1d1_pre_process.py`.

Two dataset-wide reductions are applied before phenotyping, per the
manuscript's Methods: ATC codes are abstracted to their 3rd hierarchical
level — "pharmacological subgroup", e.g. `C03C` rather than a specific
active substance like `C03CA01` (see `max_atc_level`/`atc_truncate_after_n_chars`
in `src/dim_reduce_utils.py`; note the manuscript's own worked example, "C03
for diuretics", is actually the 2nd-level therapeutic subgroup, not the 3rd
level it's introduced as — a wording imprecision in the manuscript text, not
in this code) — and any ICPC/ATC code group occurring fewer than 100 times
across the whole cohort is dropped from the candidate variable set
(`OHE_MIN_OCCURRENCES` in `src/3_pre_process.py`).

## What's not in this repo

- Real patient data, exported cohorts, or trained embedding/cluster
  artifacts — all excluded for privacy.
- Exploratory Jupyter notebooks used during analysis — excluded because
  their saved outputs contained patient-level identifiers; the reusable
  logic lives in the `.py` pipeline scripts instead.
- A Flask/MongoDB "adjudication tool" used for manual chart review during
  cohort validation — referenced by some now-removed `ops/` scripts, but its
  source was never part of this export. This is the tool behind the
  manuscript's manually-adjudicated HF outcome (GP review of full records,
  not diagnostic codes alone).
- The GPT-3.5-turbo-assisted ICPC grouping pass described in the
  manuscript's Supplementary Methods SM1 — exploratory/notebook-based, not
  part of this export; only the clinician expert-knowledge grouping
  (`5_dim_reduce.py`) is included.
- GPU-accelerated clustering via RAPIDS cuML — used opportunistically if
  importable (see `src/Top2Vec.py`), not required and not pip-installable;
  see [rapids.ai/install](https://docs.rapids.ai/install) if you want it.

## License

MIT — see [LICENSE](LICENSE). A few vendored files carry their own
upstream license; see [THIRD_PARTY_LICENSES/](THIRD_PARTY_LICENSES/).
