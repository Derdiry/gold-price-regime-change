# Phase 4 step 0: Sudan test restricted to observed days. Preregistration

Written and committed before any number on these subsets was computed. The
script `scripts/posthoc_observed_days.py` refuses to run unless this file is
committed, and copies the rules below into its results file.

## Status of this analysis

Post-hoc and diagnostic. The Sudan test set was already used once for the
Phase 3 final configurations (`results/final/test_results.json`). Nothing is
retrained, re-selected or re-weighted here. The same saved test forecasts are
scored on subsets of the same test windows, so no model, transform, ensemble
or hyperparameter choice can depend on these numbers.

## Hypothesis tested

H: *"On the Sudan track, persistence beats the models mainly because most
days are forward-filled (zero-change) days."* This is untested in the Phase 3
report. About 75% of Sudan test target days are forward-filled
(`is_ffilled == 1`); persistence is exact on a forward-filled target day when
no real observation falls between the origin and that day.

## Subsets (target days are the forecast steps t+1 … t+h of a test window)

Window counts were taken from the data flags before any forecast was scored
(`is_ffilled` only, no errors):

| h | test windows | S1: windows with ≥ 1 observed target day | S1: observed (window, step) pairs | S2: windows whose day t+h is observed | all h target days observed |
|---:|---:|---:|---:|---:|---:|
| 1 | 437 | 109 | 109 of 437 | (= S1) | 109 |
| 7 | 431 | 360 | 737 of 3017 | 106 | **0** |
| 30 | 408 | 408 | 3027 of 12240 | 104 | **0** |

- **S1, "target day observed" (primary).** Only (window, step) pairs whose
  target date is a real observation are scored. MAE, RMSE and MAPE are taken
  over those pairs. For DM, the loss of a window is the mean over its observed
  steps, and windows with no observed step are dropped
  (`diebold_mariano(..., mask=...)`).
- **S2, "h-step-ahead target day observed" (h = 7, 30).** Windows whose last
  target day t+h is observed, scored on step h only.
- **"All days in the target window observed."** 0 windows at h = 7 and h = 30,
  so it cannot be reported. At h = 1 it is identical to S1.

## Configurations and metrics

These are the same as `scripts/final_test_report.py`:
- each model under MinMax and under its selected condition, seeds 1–3;
- the MinMax ensemble and the validation-selected ensemble (saved weights);
- the Phase 2 baselines (persistence, drift, ARIMA, ETS, VECM).

Metrics:
- MAE, RMSE, MAPE;
- directional accuracy against the last observed value on non-zero true
  changes (`da_origin_nonzero`);
- MASE = MAE / MAE(persistence) on the same pairs;
- DM vs persistence, HLN-corrected, Newey–West lag h − 1, two-sided, squared
  and absolute loss;
- the number of windows and pairs.

## Verdict rule per configuration (same as Phase 3)

- **"beats persistence":** DM* < 0 and p < 0.05 in all three seeds.
- **"worse":** DM* > 0 and p < 0.05 in all three seeds.
- **"no significant difference":** otherwise.
- **Holm:** the same rule after Holm adjustment over all seed-level
  neural/ensemble tests of that loss within the subset.
- **Baselines** (one deterministic forecast): DM* sign and p < 0.05.

## Decision rule for H (fixed now)

Evaluated on **S1** with **absolute loss**, over the 18 validation-selected
configurations: 5 models' selected conditions plus the selected ensemble, at
h = 1, 7, 30. On the full test set, all 18 are "worse" under absolute loss.

- **Supported:** at most 6 of 18 remain "worse" on S1, **and** the median
  over the 18 of the seed-mean MASE is ≤ 1.00.
- **Not supported:** 12 or more of 18 remain "worse" on S1.
- **Partly supported:** anything else.

Fewer windows means less power. A configuration that is no longer "worse"
only because its p-value rose is not evidence that it is as good as
persistence. For that reason "supported" also requires the MASE condition,
which does not depend on power. The report also gives, for every
configuration, MASE on S1 next to MASE on the full test set.

S2 and squared loss are reported as secondary results and do not enter the
decision.
