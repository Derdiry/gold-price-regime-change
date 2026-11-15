# Claims and evidence

Each claim this repository makes, the result, and the generated file that contains it. `python scripts/check_claims.py` checks that every evidence file exists and that every number in a row appears in that row's evidence files.

- **Beats persistence**: the Diebold–Mariano statistic (HLN correction, Newey–West variance with lag h − 1) is negative with p < 0.05 in every seed.
- **MASE**: MAE / MAE(persistence) on the same test windows.

| ID | Claim | Result | Evidence |
|---|---|---|---|
| C1 | Headline: does any configuration beat persistence? | 231 forecast configurations; 0 beat persistence on absolute loss and 0 on squared loss | `results/summary/headline.md` |
| C2 | The Sudan test period lies outside the training range | Sudan test targets are 6.4–9.9× the training maximum (3270 SDG/g); 513 of 917 global test days are above the global training maximum (1888.70 USD/oz) | `results/summary/facts.md` |
| C3 | Scaling failure: MinMax scaling fitted on the training rows | Sudan h = 1 test MASE 31.633–259.302 across the five models; 1.643–259.302 across all settings | `results/final/test_table.md` |
| C4 | Fix: scale-free targets (log-return or RevIN), chosen per model on validation | Test MASE 1.123–12.761 on Sudan and 0.966–1.783 on the global track; PatchTST with RevIN at Sudan h = 1 remains at 12.761 | `results/final/test_table.md` |
| C5 | Best configuration overall | Global h = 30 validation-selected ensemble, MASE 0.966 (MAE 50.04 ± 0.87 vs persistence 51.80), not significant | `results/final/test_table.md`, `results/summary/headline.md` |
| C6 | Validation vs test after the regime shift (Sudan h = 30) | TFT (RevIN) and the selected ensemble have lower validation MAE than persistence (797.77) in all three seeds (784.9 / 769.6 / 777.0 and 772.7 / 759.1 / 757.2) but higher test MAE (MASE 1.522 and 1.374) | `results/final/test_table.md` |
| C7 | Statistical baselines | 0 of 21 configurations beat persistence; the closest are ETS and ARIMA at global h = 7 (MASE 0.995 each), neither significant | `results/summary/headline.md` |
| C8 | Rolling origin (5 folds, 1 seed) | Pooled over folds, 0 of 36 neural and ensemble configurations beat persistence | `results/rolling/rolling_table.md` |
| C9 | Chronos zero-shot (`chronos-t5-mini`, `chronos-t5-small`) | 0 of 12 configurations beat persistence; best MASE 1.031 (Sudan h = 30, mini); 20,456,192 and 46,154,240 parameters | `results/chronos/chronos_table.md`, `results/summary/headline.md` |
| C10 | Prediction intervals | Fixed-width 90% conformal intervals on the global track cover 81.5 / 81.1 / 76.7% for persistence and 81.2 / 80.8 / 80.2% for the selected ensemble (h = 1 / 7 / 30); Chronos 90% sample quantiles cover 42.8% of Sudan h = 1 outcomes; selected TFT, LSTM and ensemble interval widths are 0.85–1.09 × persistence's | `results/conformal/intervals.md` |
| C11 | ACLED conflict series | The monthly series built from the weekly file matches ACLED's country-month file: r = 0.9958, totals 6218 / 6217 | `results/acled/crosscheck.md` |
| C12 | ACLED conflict features, main split (lags 1–3, TFT and LSTM, 3 seeds) | No cell shows a consistent difference after Holm; before Holm, 1 of 18 (LSTM h = 7, lag 3) is worse with conflict features | `results/acled/acled_table.md` |
| C13 | ACLED conflict features, rolling check (1 seed) | Lag 2 chosen on validation (mean MAE ratio 1.0091 / 0.9866 / 1.0261 for lags 1 / 2 / 3); after Holm, on absolute loss only, conflict features hurt TFT h = 1 (also on squared loss) and help LSTM h = 1 and h = 7 (not on squared loss) | `results/acled/acled_table.md`, `results/acled/selection/lag_selection.md` |
| C14 | Regime dummies (with vs without, 3 seeds) | After Holm they help in 0 cells and hurt in 4 (absolute loss); real post-coup training rows: 21 (fold 4), 50 (fold 5), 0 (main split); TFT variable-selection weights rank `regime_2` 3/3/1 of 33 only on fold 4 h = 30 | `results/regime/regime_table.md` |
| C15 | Global → Sudan transfer (18 shared inputs, GRU and PatchTST, 3 seeds) | 0 of 6 cells show a consistent difference; freezing the first layer freezes 56,832 of 164,615 GRU weights (34.5%) but 512 of 49,223 PatchTST weights (1.0%) at h = 7 | `results/transfer/transfer_table.md` |
| C16 | Forecasting SDG/USD and rebuilding the price | 0 of 18 cells better than direct price forecasting, 2 worse; 0 beat persistence, 16 worse | `results/fx_target/fx_table.md` |
| C17 | Observed target days only (post hoc) | 11/18 selected Sudan configurations are still worse than persistence; median MASE 1.263 | `results/posthoc_observed/observed_days.md` |
| C18 | Sudan data quality, handled in code | Open 117472 on 2016-12-03 restored to 1174.72; 635 dates with several quotes (first kept, 1425 of 2528 rows); 1483 of 2908 daily rows forward-filled; 404-day gap with 403 flagged rows | `results/data/split_summary.md` |
| C19 | The Sudan series and the official exchange rate | SDG/USD from 8.38 to 539.60: 98.4% of USD value lost over 7.96 years; R² of log price on log SDG/USD 0.9976 | `results/summary/facts.md` |
| C20 | Model sizes | Trained models 34,817–224,734 parameters; PatchTST 34,817–220,894 | `results/summary/facts.md` |
| C21 | Ensembles use validation data only | Members are the top 2 by validation MAE and weights are fitted on validation predictions, stored per setting and seed | `results/summary/ensembles.md`, `tests/test_ensemble.py` |
| C22 | No target-window leakage in the scale-free transforms | Converting log-return and RevIN forecasts back to price units uses only the context window | `tests/test_transforms.py` |
