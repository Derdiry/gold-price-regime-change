# Phase 4.6: prediction intervals. Preregistration

Written and committed before any interval was scored on test.
`scripts/conformal_intervals.py` refuses to run unless this file is committed,
and copies it into its results file. Post-hoc: it uses the saved Phase 3 and
Phase 2 forecasts with no retraining. The calibration data are the validation
windows only.

## Forecasts covered

- **Phase 3 final configurations.** Each model under MinMax and under its
  selected condition, seeds 1–3; the MinMax and selected ensembles (saved
  weights).
- **Phase 2 baselines.** Persistence, drift, ARIMA, ETS, and VECM (Sudan).
  These give the reference widths.
- **Chronos (Phase 4.2), once its forecasts exist.** Its own sample quantiles,
  uncalibrated: the 5–95% interval for 90% and the 10–90% interval for 80%.
  These are scored on test with the same rules, as a separate block, so the
  neural/baseline results do not wait for Chronos.

## Split conformal

Per forecast, track, horizon h and step k = 1…h:
- calibration scores are the validation residuals, n = number of validation
  windows;
- q_k is the ⌈(n + 1)(1 − α)⌉-th smallest score;
- the test interval is formed from q_k as below.

Two score functions, both reported:
- **absolute (primary):** s = |y − ŷ|, interval ŷ ± q_k, in price units.
- **relative:** s = |y − ŷ| / P_t, where P_t is the last observed price at
  the origin (forecast-time information). The interval is ŷ ± q_k · P_t. It
  is meant to handle the level shift between validation and test.

Nominal coverage: 80% (α = 0.2) and 90% (α = 0.1).

Exchangeability between validation and test is not guaranteed: the windows
overlap in time and the level and volatility shift between the periods (most
strongly on Sudan). The tests below measure how far coverage departs from
nominal; they do not assume it holds.

## Reported on test

- Coverage: the share of (window, step) pairs inside the interval, pooled
  over steps and also per step k = 1, 7, 30 where available.
- Mean width in price units.
- Mean width relative to persistence's width under the same score and α.
- Sudan only: coverage on the S1 subset (target day observed; Phase 4
  step 0).
- Neural and ensemble values are given as mean ± std over seeds.

## Verdict per (forecast, track, h, score, α), fixed now

On the pooled test coverage (seed mean for neural/ensemble):
- **"close to nominal":** |coverage − nominal| ≤ 5 percentage points;
- **"under-covers":** coverage < nominal − 5 pp;
- **"over-covers":** coverage > nominal + 5 pp.

The tolerance is a fixed reporting threshold, not a significance test. The
windows overlap, so a binomial interval would be too narrow.
