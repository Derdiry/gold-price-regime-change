# Phase 4.3: ACLED conflict features (with vs without). Preregistration

Written and committed before any model with conflict features was trained.
`scripts/acled_report.py` refuses to run unless this file is committed and
unchanged, and copies it into its results file.

## Question

Do lagged ACLED political-violence features improve Sudan forecasts?

## Data (`src/data/acled.py`, cross-check in `results/acled/crosscheck.md`)

- **Weekly source.** ACLED "Aggregated data on Africa", country exactly
  "Sudan".
- **Political violence (author decision, definition A).** DISORDER_TYPE
  contains "Political violence", which is ACLED's own definition.
  - Against the "Number of political violence events by country-month-year"
    file, 2014-10 → 2022-12: r = 0.996, totals 6,218 vs 6,217.
  - The 3-event-type definition (Battles, Explosions/Remote violence,
    Violence against civilians) is kept as a cross-check sensitivity only.
    No models are trained with it.
- **Weekly → monthly.** Weeks start on Saturday. A week that spans two months
  is split pro rata by days.
- **Inputs.** Two features, `log1p(pv_events)` and `log1p(pv_fatalities)`,
  of calendar month month(t) − k, for one lag k ∈ {1, 2, 3} per run
  (`train.py --conflict-lag k`).
  - Only months completed before the forecast date are used. A unit test
    checks this (`tests/test_acled.py`).
  - The features are defined on every Sudan date, so the windows are
    unchanged.

## Design

- **Models.** TFT and LSTM, each under its Phase 3 validation-selected
  condition: log_return at h = 1 and 7, revin at h = 30. They are the best
  two Sudan models on Phase 3 validation MAE at every horizon. Seeds 1, 2
  and 3, paired. Hyperparameters and training are as in Phase 3.
- **Without.** The Phase 3 runs, reused because their data are byte-identical
  to the current code's, and their config and model/trainer code are
  unchanged (`results/provenance/reuse_fingerprints.md`).
- **With, main split.** 3 lags × 2 models × 3 h × 3 seeds = 54 runs, in two
  stages:
  - **Stage A** (`--val-only`): validation only; no test window is predicted.
  - **Lag selection,** for the rolling check only, on validation only.
    - The rule: take the lag with the lowest mean over (model, h, seed) of
      validation MAE(with lag k) / validation MAE(without).
    - Exact ties go to the shorter lag.
    - The selected lag and the table behind it are written to
      `results/acled/selection/lag_selection.json` before any test
      prediction of a with-run exists.
  - **Stage B:** each saved best checkpoint is scored on test once
    (`--resume`).
- **With, rolling folds 1–5** (cheap version, author decision). Only the
  selected lag, seed 1.
  - "Without" = the Phase 4.1 seed-1 runs, reused on the same byte-identity
    check.
  - Limitations as in 4.1: 1 seed per fold; the fold test blocks overlap
    the already-seen Phase 3 validation and test periods.

## Metrics

- MAE, MAPE, MASE vs persistence (seed mean ± sd), on test windows that are
  identical between with and without (checked).
- **Paired DM test** of with (model) vs without (reference), same seed: HLN
  correction, Newey–West lag h − 1, two-sided, squared and absolute loss.
  DM* < 0 means with has lower loss.

## Verdict rules

**Main split, per cell (model, h, lag) and loss:**
- **"conflict features help":** DM* < 0 and p < 0.05 in all 3 seeds;
- **"conflict features hurt":** DM* > 0 and p < 0.05 in all 3 seeds;
- **"no consistent difference":** otherwise.
- **Holm:** the same rule on Holm-adjusted p, over the 54 seed-level tests
  (18 cells × 3 seeds) per loss.
- **Summary:** the number of the 18 cells that help, hurt or show no
  consistent difference, under absolute loss (primary) and squared loss.

**Rolling folds, per (model, h) and loss, on the windows of the 5 test
blocks pooled:**
- "help" or "hurt": DM* < 0 or > 0 with p < 0.05 (1 seed);
- "no consistent difference": otherwise.
- **Holm:** over the 6 pooled tests per loss.
- **Consistency (descriptive):** folds out of 5 in which "with" has the
  lower MAE.
