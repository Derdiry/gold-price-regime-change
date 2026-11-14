# Phase 4.2: Chronos zero-shot. Preregistration

Written and committed before any Chronos forecast was scored. The scoring step
of `scripts/run_chronos.py` refuses to run unless this file is committed, and
it copies this text into its results file.

## Models

- `amazon/chronos-t5-mini` and `amazon/chronos-t5-small` from the Hugging
  Face Hub, used zero-shot: no fine-tuning, and no setting chosen on
  validation or test.
- The exact parameter count is read from the loaded model.
- Both sizes are reported. Neither is selected over the other.

## Inputs and forecasts

- Univariate, in original price units: SDG/g for Sudan, USD/oz for global.
  No scaling or transform beyond Chronos's own mean scaling.
- **Context.** The target series ending at the forecast origin, at most 512
  rows (Chronos's context length). It never crosses a row inside the 404-day
  gap, the same rule the Phase 2 statistical baselines use (there with
  750 rows). Every validation/test origin has at least 60 context rows.
- **Origins and windows.** Exactly the v2 validation and test windows used by
  every other model; the targets are checked against the Phase 2 baseline
  files. A 30-step forecast is made at every origin, and h = 1 and 7 are its
  first 1 and 7 steps.
- **Samples.** 20 sample paths (library default) with sampling seeds 1, 2, 3
  (`torch.manual_seed`).
  - **Point forecast:** the per-step median of the samples.
  - **Quantiles** 0.05, 0.10, 0.25, 0.50, 0.75, 0.90 and 0.95 are saved for
    Phase 4.6.

## Metrics

These match `scripts/final_test_report.py`:
- MAE, RMSE, MAPE;
- directional accuracy on non-zero changes;
- MASE vs persistence on the same windows;
- Diebold–Mariano vs persistence (HLN correction, Newey–West lag h − 1,
  two-sided, squared and absolute loss).

Validation metrics are reported, but no choice depends on them.

**Secondary (Sudan).** The same metrics on the S1 subset of Phase 4 step 0
(target day observed), defined in `results/posthoc_observed/PREREGISTRATION.md`.

## Verdict rule (same as Phase 3)

Per (model size, track, horizon, loss):
- **"beats persistence":** DM* < 0 and p < 0.05 in all three sampling seeds.
- **"worse":** DM* > 0 and p < 0.05 in all three sampling seeds.
- **"no significant difference":** otherwise.
- **Holm:** the same rule after Holm adjustment over all seed-level Chronos
  tests of that loss: 2 sizes × 2 tracks × 3 horizons × 3 seeds = 36 tests.
