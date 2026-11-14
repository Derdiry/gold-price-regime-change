# Project facts: gold price forecasting under regime change

Verified facts about this project. Every statement cites the file in this repository that proves it. Paths are relative to the repository root. If a fact is not in the repository, it says **NOT VERIFIED**.

---

## a) Summary

The project re-evaluates and extends Wan & Elbushra (ICCNC 2024). It tests whether deep forecasters beat a naive "no change" forecast (persistence) for two gold series: Sudanese gold in SDG/g (2015–2022) and international gold in USD/oz. Over the Sudan sample the pound lost 98.4% of its USD value. The models are LSTM, GRU, a hybrid TSMixer, PatchTST, TFT, validation-selected ensembles and zero-shot Chronos.

With a MinMax scaler fitted on training data, the models fail: Sudan test prices are 6.4–9.9× the training maximum, so test errors are 31.633–259.302× persistence at Sudan h = 1. Scale-free targets (log-returns, RevIN) largely remove that failure.

**Honest headline:** across 231 forecast configurations, none beats persistence under a Diebold–Mariano test (p < 0.05 with the favourable sign in every seed, before multiple-testing correction), on absolute or squared loss. At Sudan h = 30, validation advantages over persistence did not carry over to test.

Evidence: `README.md` (top section); `results/summary/headline.md`; `results/summary/facts.md`; `results/final/test_table.md`.

---

## b) Verified facts

| Topic | Fact | Evidence |
|---|---|---|
| Tracks | International gold (USD/oz) and Sudan local gold (SDG/g, 21k) | `docs/DETAILED_REPORT.md` §2; `configs/model_config.yaml` (`target_col`, `global_target_col`) |
| Sudan raw data | 2528 rows, 2015-01-06 → 2022-12-22. **Not included** (available on request); 20 synthetic rows show the schema | `results/summary/facts.md` line 5; `data/processed/v2/data_manifest_sudan.json` (`date_range`); `results/data/split_summary.md` line 8; `data/raw/SUDAN_DATA.md` |
| Sudan daily series | 2908 daily rows, 1425 observed, 1483 forward-filled; 404-day gap 2019-11-25 → 2021-01-02 (403 flagged rows; windows touching it dropped) | `results/summary/facts.md` line 5; `results/data/split_summary.md` lines 8–10; `data/processed/v2/data_manifest_sudan.json` |
| Sudan data quality | 635 dates with several quotes (first kept); one decimal shift (Open 117472 → 1174.72 on 2016-12-03) | `results/data/split_summary.md` line 8; `data/processed/v2/data_manifest_sudan.json` (`quality`) |
| Nature of the Sudan price | Official reference price that moves almost one-for-one with the official SDG/USD exchange rate: R² of log price on log SDG/USD 0.9976 (OLS on the 1425 observed rows), so forecasting it largely means forecasting the official exchange rate | `results/summary/facts.md` line 17; data: official reference prices for Sudanese 21-karat gold (2015–2022), obtained by the author; not redistributable; available on request for research purposes (`data/raw/SUDAN_DATA.md`) |
| Currency collapse | Official SDG/USD 8.38 (2015-01-06) → 539.60 (2022-12-22): **98.4%** of USD value lost over 7.96 years; 98.6% to the 601.70 peak (2022-03-16) | `results/summary/facts.md` lines 9–10 |
| Out-of-range test | Sudan test targets 20818–32525 SDG/g = **6.4–9.9×** the training maximum (3270). Global: 513 of 917 test days above the training maximum 1888.70 USD/oz | `results/summary/facts.md` lines 21–22 |
| Global period | Training windows from 2010-08 (earlier rows have undefined features); test 2021-05-10 → 2024-12-30 | `docs/DETAILED_REPORT.md` §3.1; `docs/implementation_notes.md` D3 |
| Models | LSTM, GRU, TSMixer (**hybrid**: LSTM(64) + Conv1D(64) + MaxPooling1D added to a modified copy of Google Research's TSMixer), PatchTST, TFT; validation-selected ensemble (top-2 by validation MAE, SLSQP weights fitted on validation); zero-shot Chronos `amazon/chronos-t5-mini` and `amazon/chronos-t5-small` | `src/models/{lstm,gru,tsmixer,patchtst,tft}.py`; `reference/tsmixer_paper_original.py` (header and `res_block`); `src/models/ensemble.py`; `scripts/run_chronos.py` (`SIZES`) |
| Baselines | Persistence, drift, ARIMA and ETS on log price, VECM (Sudan, log price / intl price / FX) | `src/evaluation/baselines.py` |
| Window, horizons, split | 60-day input window; horizons 1, 7, 30; chronological 70/15/15 by rows | `configs/model_config.yaml` (`sequence`, `split`) |
| Inputs | 33 features (Sudan), 29 (global) | `docs/DETAILED_REPORT.md` §3.1 |
| Seeds | 3 seeds (1, 2, 3) on the main split and in the Phase 4 main-split experiments | `configs/model_config.yaml` (`experiment.seeds`) |
| Rolling origin | 5 expanding folds, **1 seed** (seed 1); folds overlap the main validation and test periods | `src/data/folds.py` (`N_BLOCKS = 6`); `results/rolling/PREREGISTRATION.md` |
| Window counts | Sudan validation 223 / 217 / 194, test 437 / 431 / 408; global test 917 / 911 / 888 (h = 1 / 7 / 30) | `results/data/split_summary.md` |
| Statistical test | Diebold–Mariano vs persistence on absolute and squared loss; Newey–West variance (Bartlett, lag h − 1); HLN small-sample correction; two-sided; Holm adjustment over seed-level tests per family | `src/evaluation/stats.py` (`diebold_mariano`, `holm`); `tests/test_stats.py` |
| MinMax failure | MinMax scaling fitted on the training rows, Sudan h = 1 test MASE **31.633** (TSMixer) to **259.302** (TFT); all settings 1.643–259.302 | `results/final/test_table.md` §"sudan h=1"; `docs/CLAIMS.md` C3 |
| After the fix | Validation-selected condition: Sudan MASE 1.123–12.761, global 0.966–1.783. Best per Sudan horizon: 1.392 (h=1, TFT log-return), 1.123 (h=7, ensemble), 1.324 (h=30, LSTM RevIN). **PatchTST RevIN, Sudan h = 1, still 12.761×** | `results/final/test_table.md`; `docs/CLAIMS.md` C4 |
| Headline | **231** forecast configurations, **0** beat persistence on absolute loss, **0** on squared loss (before multiple-testing correction; configurations overlap between experiments, so 231 counts evaluations). Best seed-mean MASE 0.966 (global h = 30 ensemble), not significant | `results/summary/headline.md` line 17 and table; `docs/CLAIMS.md` C1, C5 |
| Statistical baselines | 0 of 21 configurations beat persistence; closest ETS global h = 7, MASE 0.995 | `results/summary/headline.md`; `docs/DETAILED_REPORT.md` §4 |
| Persistence test MAE | Sudan 88.00 / 297.56 / 890.35 SDG/g; global 13.72 / 27.01 / 51.80 USD/oz | `results/final/test_table.md`; `docs/DETAILED_REPORT.md` §4 |
| Validation vs test, Sudan h = 30 | Persistence validation MAE 797.77, test 890.35. TFT (RevIN) validation 784.9 / 769.6 / 777.0, test 1411.6 / 1448.7 / 1205.8. Selected ensemble validation 772.7 / 759.1 / 757.2, test 1196.6 / 1351.3 / 1122.9. Better than persistence on validation in 3/3 seeds, worse on test in 3/3. Test absolute loss significantly worse before Holm, not after Holm | `results/final/test_table.md` §"Sudan h=30 (validation advantage check)", lines 154, 167, 169 |
| Conformal, global, fixed-width (absolute score) 90% | Persistence 81.5 / 81.1 / 76.7% (h = 1 / 7 / 30); all statistical baselines 76.3–81.5%; selected ensemble 81.2 / 80.8 / 80.2%; all validation-selected neural models 73.4–83.0% | `results/conformal/intervals.md` §"global h=1/7/30", column "abs 90% cov" |
| Conformal, global, price-scaled (relative score) 90% | Selected models and ensemble 82.9–91.4%; persistence 88.8 / 88.4 / 86.5% | `results/conformal/intervals.md`, column "rel 90% cov" |
| Chronos intervals | Uncalibrated 5–95% sample quantiles cover **42.8%** of Sudan h = 1 outcomes (mini 42.8 ± 3.2, small 42.8 ± 0.2) | `results/conformal/intervals.md` lines 33–34 |
| Chronos accuracy | 0 of 12 configurations beat persistence; best MASE 1.031 (Sudan h = 30, mini) | `results/chronos/chronos_table.md`; `docs/CLAIMS.md` C9 |
| Parameter counts | Trained models **34,817–224,734** (PatchTST 34,817–220,894); Chronos mini **20,456,192**, small **46,154,240** | `results/summary/facts.md` lines 27–28 |
| Run records | 675 saved training-run records (210 in Phase 3) | `results/summary/facts.md` line 26 |
| ACLED | Monthly series cross-check vs ACLED's country-month file: r = 0.9958, totals 6218 / 6217. Main split (3 seeds): 0 of 18 cells improve; 1 worse before Holm (LSTM h = 7, lag 3), 0 after. Rolling (1 seed, lag 2): helps LSTM h = 1 and h = 7, hurts TFT h = 1 after Holm | `results/acled/crosscheck.md`; `results/acled/acled_table.md`; `docs/CLAIMS.md` C11–C13 |
| Regime dummies | After Holm: help 0 cells, hurt 4 (absolute loss). Real post-coup training rows: 21 (fold 4), 50 (fold 5), 0 (main split). TFT variable-selection weights rank `regime_2` 3/3/1 of 33 on fold 4 h = 30 only | `results/regime/regime_table.md`; `docs/CLAIMS.md` C14 |
| Transfer | Global → Sudan vs scratch (18 shared inputs, GRU and PatchTST, 3 seeds): 0 of 6 cells differ consistently | `results/transfer/transfer_table.md`; `docs/CLAIMS.md` C15 |
| FX target | 0 of 18 cells better than direct price forecasting, 2 worse; 0 beat persistence, 16 worse | `results/fx_target/fx_table.md`; `docs/CLAIMS.md` C16 |
| Observed days only (post hoc) | 11 of 18 selected configurations still worse than persistence (18/18 on the full test set); median MASE 1.263 | `results/posthoc_observed/observed_days.md`; `docs/CLAIMS.md` C17 |
| Design decisions | Scale-free targets because the Sudan test range is outside the training range; validation-only selection and ensembles; windows with undefined features dropped, never back-filled; windows touching the 404-day gap dropped | `docs/implementation_notes.md` D1–D5 |
| Tests | `pytest`: **42 passed, 1 skipped, 0 failed** in this repository without the Sudan file. The skip is `tests/test_quality.py:35`, "raw Sudan file not present" | run `python -m pytest -q`; `tests/` |
| Tracking | MLflow tracking of every run (local `mlruns/` store, not distributed) | `train.py` (imports `mlflow`, `src.training.mlflow_utils`) |
| Release checks | `scripts/release_checks.py`: secrets, personal paths, restricted data, compile, ensembles, baselines, claims, all PASS | run `python scripts/release_checks.py . --allow-git` |

---

## c) What was preregistered and what was not

| Experiment | Preregistered? | Evidence |
|---|---|---|
| Phase 2 statistical baselines | **No** | no preregistration file in `results/baselines/` |
| Phase 3 condition selection (validation only) | Rule fixed in `scripts/select_transforms.py`, committed (`3345372`) before any test prediction. Not a `PREREGISTRATION.md` | `docs/DETAILED_REPORT.md` §3.4 |
| **Phase 3 main test comparison** (5 models × conditions × 3 horizons × 2 tracks, 3 seeds) | **No.** Its comparison script (`90fd2ba`, 22:02:03) was committed 6 min 18 s after the stage-2 test run started (`6bb65f5`, 21:55:45, 2026-10-28). The repository cannot show when results were first read | `docs/DETAILED_REPORT.md` §3.4; `results/provenance/stage2_launch.md`; `docs/implementation_notes.md` "Phase 3 stage 2 provenance" |
| 4.1 Rolling origin | Yes, `0d36807`, before training | `results/rolling/PREREGISTRATION.md` |
| 4.2 Chronos | Yes, `9122d6d`, before prediction and scoring | `results/chronos/PREREGISTRATION.md` |
| 4.3 ACLED | Yes, `205af44`, before training; lag chosen on validation before any test prediction | `results/acled/PREREGISTRATION.md`; `results/acled/selection/lag_selection.md` |
| 4.4 Regime dummies | Yes, `205af44`, before training | `results/regime/PREREGISTRATION.md` |
| 4.5 Transfer | Yes, `205af44`, before training | `results/transfer/PREREGISTRATION.md` |
| 4.6 Conformal intervals | Yes, `8874076`, before scoring | `results/conformal/PREREGISTRATION.md` |
| 4.7 FX target | Yes, `2fcd848`, before training | `results/fx_target/PREREGISTRATION.md` |
| Observed-days diagnostic (post hoc, on the already-used test set) | Decision rule committed (`c954d71`) before its numbers were computed | `results/posthoc_observed/PREREGISTRATION.md` |
| Rolling fold-1 check | **No** (post hoc) | `results/rolling/fold1_sudan_h30_check.md` line 3 |

The commit hashes above refer to the original development history, which is not published. Inside each report they are embedded as `preregistration_commit` (for example `results/acled/acled_table.md` line 3). The Phase 4 report scripts refuse to run unless their preregistration is committed and unchanged (`scripts/paired_common.py`, `require_committed`).

---

## d) Limitations

- One dataset (two series), one chronological split, so a **single test period** per track (Sudan 2021-10-12 → 2022-12-22; global 2021-05-10 → 2024-12-30). `docs/DETAILED_REPORT.md` §3.1, §8.
- **Small Sudan validation set:** 223 / 217 / 194 windows, all in 2021-03 → 2021-10. `README.md` "Limitations"; `results/data/split_summary.md`.
- **CPI/M2 look-ahead (global track):** `clean_macro` (`src/data/processors.py:48`) time-interpolates and back-fills monthly CPI and M2, with no publication lag. Not corrected or measured. `README.md` "Limitations".
- Rolling origin: one seed per fold; the folds overlap the main validation and test periods. `README.md` "Limitations".
- Regime test has low power: 21 / 50 real post-coup training rows (folds 4 / 5), 0 on the main split. `results/regime/regime_table.md`.
- ACLED rolling check: one seed. ACLED and Yahoo/FRED data cannot be redistributed. `README.md` "Limitations".
- Hyperparameters fixed in advance and not tuned (TFT batch size 8, set for memory). `README.md` "Limitations"; `docs/implementation_notes.md` D12.
- Phase 3 main comparison not preregistered (§c).
- The Sudan series is an official reference price that moves almost one-for-one with the official exchange rate and is forward-filled on 1483 of 2908 days, so the task largely means forecasting the official exchange rate. `results/summary/facts.md`.
- The Sudan data are not public. Sudan results can be reproduced only with the full file (available on request). `data/raw/SUDAN_DATA.md`.

---

## e) Related publication

Wan, Y., & Elbushra, M. (2024). Forecasting Sudan Gold Prices with a Hybrid Deep Learning Approach. *2024 International Conference on Cloud and Network Computing (ICCNC)*, pp. 71–78. doi:10.1109/ICCNC63989.2024.00020

This repository re-evaluates and extends that work. Its dataset is the paper's dataset (not included here). The paper's hybrid TSMixer code is `reference/tsmixer_paper_original.py`. The citation text is in `README.md` ("Citation") and `CITATION.cff`. The paper itself is not in the repository, so its contents are **NOT VERIFIED** here.
