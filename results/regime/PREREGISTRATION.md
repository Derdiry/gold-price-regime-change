# Phase 4.4: regime dummies (with vs without). Preregistration

Written and committed before any no-regime model was trained.
`scripts/regime_report.py` refuses to run unless this file is committed and
unchanged, and copies it into its results file.

## Question

Do the political-regime dummies `regime_1` (transitional, from 2019-04-11)
and `regime_2` (post-coup, from 2021-10-25) improve Sudan forecasts?

## Design

- **Models.** TFT and LSTM, each under its Phase 3 validation-selected
  condition: log_return at h = 1 and 7, revin at h = 30. They are the two best
  Sudan models on Phase 3 validation MAE at every horizon
  (`results/selection/selection.json`). No test result was used.
- **Without.** The 33 standard Sudan inputs minus `regime_1` and `regime_2`,
  so 31 inputs (`train.py --ablation no_regime`). Hyperparameters, epochs,
  early stopping and batch sizes are as in Phase 3.
- **Seeds.** 1, 2, 3, paired: each "without" run is compared with the "with"
  run of the same seed.
- **Horizons.** 1, 7 and 30.
- **Family F (rolling folds 4 and 5).** These are the only folds with
  post-coup rows in training (`src/data/folds.py`).
  - "With" = the Phase 4.1 seed-1 runs, plus seeds 2 and 3 trained now under
    the same settings.
  - The 4.1 runs are reused because their data are byte-identical to the
    current code's, and their config and model/trainer code are unchanged
    (`results/provenance/reuse_fingerprints.md`).
- **Family M (main split)** (author addition).
  - "With" = the Phase 3 runs, seeds 1–3, which are reused on the same
    byte-identity check.
  - "Without" = new runs.

### Regime rows (real = not forward-filled and not in the 404-day gap)

| split | `regime_1` = 1 rows (real) | `regime_2` = 1 rows (real) |
|---|---|---|
| fold 4 train (2015-01-06 → 2022-01-26) | 928 (230) | **94 (21)** |
| fold 4 val / test | 0 / 0 | 110 (29) / 110 (24) |
| fold 5 train (2015-01-06 → 2022-05-16) | 928 (230) | **204 (50)** |
| fold 5 val / test | 0 / 0 | 110 (24) / 110 (32) |
| main train (2015-01-06 → 2020-08-01) | 479 (153) | **0 (0)** |
| main val (2020-08-02 → 2021-10-11) | 436 (74) | 0 (0) |
| main test (2021-10-12 → 2022-12-22) | 13 (3) | 424 (106) |

**Stated in advance:**
- On folds 4 and 5, "with" models learn `regime_2` from only 21 and 50 real
  observations. This test therefore has **low power**. A null result there
  is not evidence that the regime has no effect.
- On the main split, `regime_2` is 0 on every training and validation row
  and 1 on almost every test row. The models never see it vary before test,
  so family M measures the effect of an input that is constant in training
  and switches on at test. It does not measure a learned regime effect.

## Metrics

On each run's own test windows. These are identical between "with" and
"without", which the report checks from the saved origins and targets.

- MAE, MAPE, MASE vs persistence (seed mean ± sd).
- **Paired DM test** of "with" (model) vs "without" (reference): HLN
  correction, Newey–West lag h − 1, two-sided, squared and absolute loss.
  DM* < 0 means "with" has lower loss.

## Verdict rules

**Per cell (model, fold or main, h) and loss:**
- **"regime dummies help":** DM* < 0 and p < 0.05 in all 3 seeds;
- **"regime dummies hurt":** DM* > 0 and p < 0.05 in all 3 seeds;
- **"no consistent difference":** otherwise.

**Holm:** the same rule on Holm-adjusted p.
- Family F: over its 36 seed-level tests (12 cells × 3 seeds) per loss.
- Family M: over its 18 tests (6 cells × 3 seeds) per loss.

**Summary:** per family, the number of cells that help, hurt or show no
consistent difference under absolute loss (primary) and squared loss. The
fold summary is reported with the low-power caveat above.

## TFT variable importance

TFT variable-selection weights, averaged over each run's validation windows,
for the "with" runs (seeds 1–3) on folds 4 and 5 and on the main split:
- the weight and the rank (out of 33) of `regime_1` and `regime_2`;
- the top 5 inputs.

This is descriptive. No verdict is drawn from it.
