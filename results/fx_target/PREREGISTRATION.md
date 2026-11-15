# Phase 4.7: FX as the target. Preregistration

Written and committed before any FX-target model was trained or scored.
`scripts/fx_target_report.py` refuses to run unless this file is committed,
and copies it into its results file.

## Note

One section was removed from this file for the public release for data-confidentiality reasons, and one sentence below was shortened; the rest is unchanged.

## Models and protocol (only the target changes)

- **Target.** Sudan track, `fx_rate` (SDG/USD) at t+1 … t+h, for
  h = 1, 7, 30.
- **Inputs.** The same 5 models and the same input features. The technical
  indicators are still computed on the price, so the inputs are identical to
  the Phase 3 price runs.
- **Same as Phase 3:** windows, splits, hyperparameters, epochs, early
  stopping and batch sizes.
- **Transform conditions** minmax, log_return and revin, with the forecast-time
  inverse based on the last observed FX value. Under log_return, the
  price-relative inputs are divided by the last FX value, because the
  transform code uses the target's last value.
- **Isolation.** Runs go under `results/fx_target/`, `logs/v2/fx_target/` and
  the MLflow experiment `gold-sudan-v2-fx-target`. They are not registered.

## Reconstruction to price (forecast-time information only)

- **Main (a forecast):**
  `price_hat_{t+k} = P_t × fx_hat_{t+k} / fx_t`. P_t and fx_t are the last
  observed price and FX at the origin. FX persistence therefore rebuilds to
  price persistence.
- **Diagnostic (uses future data, not a forecast):**
  `price_oracle_{t+k} = P_t × (fx_hat_{t+k} / fx_t) × (intl_{t+k} / intl_t)`.
  It uses the realised future international price. It shows how much of the
  error comes from FX alone. It is never compared with the forecasts as if it
  were one, and it does not enter any verdict.

## Selection (validation only; same rule as Phase 3)

Seed 1, all 3 conditions. For each (model, h), choose the condition with the
lowest validation MAE of the **rebuilt price** (main reconstruction). If
conditions are within 1%, choose the simplest, in the order
minmax < log_return < revin. Then run seeds 2 and 3 for the chosen condition;
seed 1 is scored on test from its saved checkpoint. A validation-selected
ensemble is formed the same way as in Phase 3: top-2 by validation MAE of the
rebuilt price, SLSQP weights fitted on validation.

## Test evaluation (once per final configuration)

In price units (SDG/g), on the Phase 3 test windows: MAE, RMSE, MAPE,
directional accuracy on non-zero changes, and MASE vs persistence. The FX
models' errors in SDG/USD are also reported.

Comparisons:
1. **FX route vs persistence:** DM, HLN, Newey–West lag h − 1, squared and
   absolute loss.
2. **FX route vs the direct price route:** same model, its Phase 3 selected
   condition, matched seed (FX seed s vs price seed s). The ensembles are
   compared as selected FX ensemble vs selected price ensemble.

## Verdict rules

**Per configuration and loss:**
- **vs persistence:** "beats persistence" / "worse" if the sign agrees and
  p < 0.05 in all three seeds; otherwise "no significant difference".
- **vs direct:** "FX route better" / "FX route worse" if the sign agrees and
  p < 0.05 in all three matched seed pairs; otherwise "no significant
  difference".
- **Holm:** the same rules after Holm adjustment over all seed-level tests of
  that comparison and loss (6 configurations × 3 h × 3 seeds = 54 tests).

**Summary answer**, from absolute loss vs direct, over the 18
(configuration, h) cells:
- "the FX route improves on direct price forecasting" if at least 10 of 18
  are "FX route better" and none is "FX route worse";
- "the FX route is worse" if at least 10 of 18 are "FX route worse" and none
  is "better";
- otherwise "no consistent difference".

**Secondary**, outside the verdicts: the same metrics on the S1 subset (target
day observed; Phase 4 step 0), and the oracle diagnostic.
