# Reproducibility for Imputation Disagreement for Selective Sepsis Prediction in Two Hospitals

This repository contains the analysis code, fixed experiment settings, and aggregate evidence for the retrospective PhysioNet Challenge 2019 analysis. The frozen analysis uses cohort version `a-data-v1-20261104-r1`, main run `20261105T055420_02d05b15`, and sensitivity run `20261105T055445_54dd08de`. Patient records are obtained directly from [PhysioNet Challenge 2019, version 1.0.0](https://doi.org/10.13026/v64v-d857).

## Reproduce the primary analysis

Use Python 3.14 and install the recorded dependency versions from the repository root.

```sh
python -m pip install -r requirements-lock.txt
python -m pytest -q
```

Use the [official PhysioNet download instructions](https://physionet.org/content/challenge-2019/1.0.0/#files) to obtain the `training/training_setA` and `training/training_setB` folders. With the AWS CLI, the following downloads the two sets anonymously from the PhysioNet Open bucket.

```sh
aws s3 sync --no-sign-request s3://physionet-open/challenge-2019/1.0.0/training/ .local/raw
python scripts/rebuild_cohort.py --raw .local/raw
python scripts/reproduce_primary.py
```

The builder verifies the names and bytes of all 40,336 raw records, reconstructs the 24,095-person cohort and the five fixed partitions, and checks their digests against the frozen input. It accepts either the two directories of `.psv` files or locally created `training_setA.zip` and `training_setB.zip` archives with each site's `.psv` files at the archive root.

The primary replay fits the seed-2026 logistic model from scratch. It checks 12 rows of metrics, four paired selective-risk intervals, the internal and external AUROC intervals, and the paired dispositions of positive patients. Expected results include 0/8 internal and 0/38 external positive-set coverage. At the matched 60% target, the B4 minus B2 selective-risk differences are +0.109 percentage points internally and −0.036 percentage points externally.

## Full experiment and aggregate evidence

```sh
python scripts/run_experiments.py --config configs/main.yaml
python scripts/run_experiments.py --config configs/supplements.yaml --rf --m-sensitivity --alpha-sensitivity
```

The main configuration covers five splits and scenarios S0–S4. The supplement configuration covers the random-forest, imputation-count, and conformal-level sensitivity analyses. Each execution writes a new run under `.local/results/clinical/`. Keep `.local/` outside version control because it contains patient-level inputs and outputs.

`evidence/cohort/` contains aggregate eligibility, event, and missingness counts. `evidence/main/` and `evidence/supplements/` contain the reference metrics, bootstrap intervals, method status, coverage, and stability summaries. `thresholds.csv` contains numerical cutoff summaries, with tie breakers regenerated from local inputs. The main aggregate evidence also records the post hoc random-allocation reference and paired counts. Positive events are sparse, and the evidence supports descriptive reliability evaluation within the reported scope.
