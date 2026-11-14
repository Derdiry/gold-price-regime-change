# Phase 4.1: rolling-origin evaluation. Preregistration

Written and committed before any fold model was trained or any fold forecast
was scored. `scripts/rolling_report.py` refuses to run unless this file is
committed, and copies it into its results file.

## Folds (`src/data/folds.py`, unit-tested in `tests/test_folds.py`)

- **Evaluation period.** The v2 validation + test period from the first row
  where a complete window exists (Sudan: the end of the 404-day gap plus
  60 rows).
- **Blocks.** The period is cut into 6 equal consecutive row blocks
  B0 … B5. Fold k = 1 … 5 uses:
  - train: all rows before B_{k−1} (expanding);
  - val: B_{k−1}, used for early stopping and ensemble weights;
  - test: B_k.
- **Windows.** Built as in the main protocol: targets never cross a split
  boundary, and windows touching an invalid row are dropped. Counts were taken
  before training.

| track | fold | train rows end | val | test | test windows h=1 / 7 / 30 |
|---|---:|---|---|---|---|
| sudan | 1 | 2021-03-02 | 2021-03-03 → 2021-06-20 | 2021-06-21 → 2021-10-08 | 110 / 104 / 81 |
| sudan | 2 | 2021-06-20 | 2021-06-21 → 2021-10-08 | 2021-10-09 → 2022-01-26 | 110 / 104 / 81 |
| sudan | 3 | 2021-10-08 | 2021-10-09 → 2022-01-26 | 2022-01-27 → 2022-05-16 | 110 / 104 / 81 |
| sudan | 4 | 2022-01-26 | 2022-01-27 → 2022-05-16 | 2022-05-17 → 2022-09-03 | 110 / 104 / 81 |
| sudan | 5 | 2022-05-16 | 2022-05-17 → 2022-09-03 | 2022-09-04 → 2022-12-22 | 110 / 104 / 81 |
| global | 1 | 2017-09-15 | 2017-09-18 → 2018-12-03 | 2018-12-04 → 2020-02-24 | 306 / 300 / 277 |
| global | 2 | 2018-12-03 | 2018-12-04 → 2020-02-24 | 2020-02-25 → 2021-05-10 | 305 / 299 / 276 |
| global | 3 | 2020-02-24 | 2020-02-25 → 2021-05-10 | 2021-05-11 → 2022-07-26 | 305 / 299 / 276 |
| global | 4 | 2021-05-10 | 2021-05-11 → 2022-07-26 | 2022-07-27 → 2023-10-12 | 306 / 300 / 277 |
| global | 5 | 2022-07-26 | 2022-07-27 → 2023-10-12 | 2023-10-13 → 2024-12-30 | 305 / 299 / 276 |

Sudan folds 4 and 5 train on data after the 2021-10-25 coup; fold 3 has it in
its validation block.

## What is run

- **Neural.** Each of the 5 models under its Phase 3 validation-selected
  condition (`results/selection/selection.json`), seed 1. Hyperparameters,
  epochs, early stopping and batch sizes are as in Phase 3. Scalers and
  window scalers are fitted on each fold's training rows.
- **Ensemble.** Per fold: top-2 of the 5 models by fold-validation MAE, with
  SLSQP weights fitted on fold validation.
- **Baselines.** Persistence, drift, ARIMA, ETS, and VECM (Sudan). Their
  settings are chosen per fold on fold-validation MAE with the Phase 2 grids,
  and they are refit at every origin (`scripts/run_baselines.py --fold k`).

## Limitations, stated in advance

- **One seed per fold** (author decision). Phase 3 showed that seed spread can
  be large, so a single fold result is noisy. The pooled verdict below
  averages over folds, not over seeds.
- **The folds are not untouched data.** The transform per model was chosen on
  the Phase 3 validation period, which lies inside the Sudan fold-1 test block
  and the global fold-1 and fold-2 test blocks. The remaining fold test blocks
  lie inside the Phase 3 test period, whose results were already seen. No
  choice is made from the fold results.

## Metrics

Per fold, and pooled over the 5 test blocks (windows concatenated in time
order):
- MAE, MAPE;
- MASE vs persistence on the same windows;
- directional accuracy on non-zero changes;
- DM vs persistence (HLN, Newey–West lag h − 1, two-sided, squared and
  absolute loss).

Test blocks do not share target days, so lag h − 1 still covers every
overlap.

## Verdict rules

**Per (configuration, track, h, loss), on the pooled windows:**
- **"beats persistence":** DM* < 0 and p < 0.05;
- **"worse":** DM* > 0 and p < 0.05;
- **"no significant difference":** otherwise.
- **Holm:** the same after Holm adjustment over the 36 pooled neural/ensemble
  tests of that loss (6 configurations × 2 tracks × 3 h).

**Consistency (descriptive):** the number of folds out of 5 with MASE < 1.

**Summary:** the list of (configuration, track, h) that beat persistence on
the pooled windows under absolute loss, with and without Holm.
