# Detailed report

The full audit behind the [README](../README.md): data, protocol, baselines, every result table, the follow-up experiments and their interpretation. Limitations, data availability and how to run the code are in the README.

## Contents

1. [Reviewer guide](#1-reviewer-guide)
2. [Data](#2-data)
3. [Protocol](#3-protocol)
4. [Baselines](#4-baselines)
5. [Results](#5-results)
6. [Ablations and further experiments](#6-ablations-and-further-experiments)
7. [Interpretation](#7-interpretation)
8. [Future work](#8-future-work)

---

## 1. Reviewer guide

These files carry most of the research content, in suggested reading order.

| File | What it shows |
|---|---|
| [`results/summary/headline.md`](../results/summary/headline.md) | One table answering the main question: per experiment, how many configurations beat persistence, and the best MASE. |
| [`src/data/transforms.py`](../src/data/transforms.py), [`tests/test_transforms.py`](../tests/test_transforms.py) | The scale fix: log-return and RevIN conditions. Every statistic comes from the context window. `test_inverse_ignores_target_window` and the two `*_uses_only_*` tests prove that converting back to price units never reads the target window. |
| [`src/data/processors.py`](../src/data/processors.py), [`tests/test_windows.py`](../tests/test_windows.py) | Chronological split, scalers fitted on training rows, and window building. Windows that touch the 404-day gap are dropped (`test_windows_touching_long_gap_are_dropped`). |
| [`src/evaluation/stats.py`](../src/evaluation/stats.py), [`tests/test_stats.py`](../tests/test_stats.py) | Diebold–Mariano with the HLN correction and Newey–West variance (lag h − 1), MASE, and Holm adjustment. |
| [`src/evaluation/metrics.py`](../src/evaluation/metrics.py), [`src/evaluation/baselines.py`](../src/evaluation/baselines.py) | Metrics in price units. Directional accuracy is also reported on non-zero changes only, with its attainable ceiling. Persistence, drift, ARIMA/ETS on log price, and the Sudan VECM. |
| [`train.py`](../train.py) | The training entry point: validation-only model selection, the test scored once per final configuration, and provenance (commit and full config) in every run record. |
| `results/*/PREREGISTRATION.md` | The verdict rule of each Phase 4 experiment and of the post-hoc diagnostic, committed before its test numbers. Phase 3 has none ([§3.4](#34-preregistration-and-test-use)). |
| [`results/final/test_table.md`](../results/final/test_table.md) | Phase 3: every model × condition × horizon × track against all baselines. |
| [`scripts/make_tables.py`](../scripts/make_tables.py), [`scripts/compare_regenerated.py`](../scripts/compare_regenerated.py) | Regenerate every table from saved files, and check the result against the committed tables. |
| [`docs/CLAIMS.md`](CLAIMS.md), [`docs/implementation_notes.md`](implementation_notes.md) | The claims with their evidence; design decisions and run provenance. |

Model code: [`src/models/patchtst.py`](../src/models/patchtst.py) implements channel independence by folding channels into the batch dimension. [`src/models/tft.py`](../src/models/tft.py) has variable selection and attention; the TFT feature rankings reported in `results/` are its variable-selection weights, not attention. [`src/models/tsmixer.py`](../src/models/tsmixer.py) is the author's hybrid TSMixer from the earlier paper, not a reproduction of the original TSMixer: it adds LSTM(64), Conv1D(64) and MaxPooling1D layers to each residual block of Google Research's TSMixer. The paper's code, a modified copy of Google Research's Apache-2.0 file, is in [`reference/`](../reference/).

---

## 2. Data

| Track | Source | In this repository? |
|---|---|---|
| Sudan | `data/raw/new_Data.csv`, the author's dataset from the earlier published study (official reference prices for Sudanese 21-karat gold, obtained by the author): 2528 rows, 2015-01-06 → 2022-12-22, with the local price, the international price, the 21k price per gram, SDG/USD and international OHLC/volume | **No.** Available on request. [`data/raw/SUDAN_DATA.md`](../data/raw/SUDAN_DATA.md) describes it, and `data/raw/new_Data_SYNTHETIC_sample.csv` holds 20 **synthetic** rows in the same schema |
| Sudan | `data/processed/v2/sudan_merged.csv`: the cleaned daily series with features, derived only from `new_Data.csv` | No (rebuilt from `new_Data.csv` by `src/data/pipeline.py`) |
| Global | Yahoo Finance (gold futures, GLD, UUP, BNO) and FRED (DGS10, CPI, M2), as a 2026-04-04 snapshot | No: third-party terms. It can be rebuilt with `src/data/fetchers.py` (`FRED_API_KEY` needed), but the result will not be byte-identical |
| Sudan | ACLED weekly *Aggregated data on Africa* and the *political violence events by country-month-year* files | No: ACLED terms. The user downloads them into `data/raw/sudan/` (gitignored) |

**Price and exchange rate.** The Sudan series is an official reference price that moves almost one-for-one with the official SDG/USD exchange rate (R² of log price on log exchange rate = 0.9976, OLS on the 1425 observed rows), so forecasting it largely means forecasting the official exchange rate. Data: official reference prices for Sudanese 21-karat gold (2015–2022), obtained by the author; not redistributable; available on request for research purposes. Evidence: [`results/summary/facts.md`](../results/summary/facts.md).

**Data-quality issues, handled in code; the raw file is unchanged.**
- **Decimal shift.** Open = 117472 on 2016-12-03 is restored to 1174.72.
- **Duplicate dates.** 635 dates have several quotes. The first quote of each day is kept, giving 1425 of the 2528 rows.
- **Forward-filled target.** Daily resampling forward-fills 1483 of 2908 days. An `is_ffilled` flag column marks them.
- **The 404-day gap** (2019-11-25 → 2021-01-02). Its 403 rows are flagged, and every window whose context or target touches it is dropped.

Evidence: [`results/data/split_summary.md`](../results/data/split_summary.md), [`docs/implementation_notes.md`](../docs/implementation_notes.md).

**Exchange rate.** SDG/USD (the official rate) moves from 8.38 on 2015-01-06 to 539.60 on 2022-12-22, with a peak of 601.70 on 2022-03-16. Two political breaks fall inside the sample: 2019-04-11 (`regime_1`) and the 2021-10-25 coup (`regime_2`).

---

## 3. Protocol

### 3.1 Splits and windows
- **Split.** Chronological 70/15/15 by rows.
- **Sudan:** train 2015-01-06 → 2020-08-01, validation 2020-08-02 → 2021-10-11, test 2021-10-12 → 2022-12-22. Test windows number 437 / 431 / 408 (h = 1 / 7 / 30).
- **Global:** the split by rows is train → 2017-09-15, validation 2017-09-18 → 2021-05-07, test 2021-05-10 → 2024-12-30. Training windows start in 2010-08, because earlier rows have undefined features (implementation note D2). Test windows number 917 / 911 / 888.
- **Inputs.** Context length is 60 days. Inputs are 33 features (Sudan) and 29 (global).
- **Out-of-range test periods.** Every Sudan test target is above the training maximum of 3270 SDG/g, at 6.4–9.9× that maximum. On the global track, 513 of 917 test days close above the training maximum of 1888.70 USD/oz.

### 3.2 Conditions and selection
- **Conditions.** `minmax` (MinMax scaling fitted on the training rows), `log_return` and `revin` ([`src/data/transforms.py`](../src/data/transforms.py)). Hyperparameters, epoch budgets and early stopping are the same for every condition and were not tuned. The TFT batch size is 8, set for memory (implementation note D12).
- **Selection rule.** For each (model, track, horizon), pick the condition with the lowest seed-1 validation MAE in price units. Conditions within 1% count as tied, and the tie goes to the simpler one ([`results/selection/validation_table.md`](../results/selection/validation_table.md)).
- **Ensemble.** Top-2 of the selected configurations by validation MAE, with SLSQP weights fitted on validation predictions ([`results/summary/ensembles.md`](../results/summary/ensembles.md)).

### 3.3 Comparison with persistence
- Everything is scored on the same test windows, in price units.
- **MASE** = MAE / MAE(persistence).
- **Test.** Diebold–Mariano on absolute and on squared loss, with the HLN small-sample correction, Newey–West variance with lag h − 1, two-sided.
- **Verdict.** "Beats" or "worse" requires the sign to agree and p < 0.05 in all three seeds. Holm adjustment is applied over the seed-level tests of each family.

### 3.4 Preregistration and test use

The test set is evaluated once per final configuration. Each Phase 4 experiment's verdict rule, and that of the post-hoc diagnostic, is in a `PREREGISTRATION.md` committed before its test predictions were scored. The Phase 3 main comparison was not preregistered (rows 1–2):

| Experiment | Rule file | Commit | Committed before |
|---|---|---|---|
| Phase 3 selection | `scripts/select_transforms.py` (rule fixed before the runs) | `3345372` | any test prediction (seed-1 runs were validation-only) |
| Phase 3 test comparison (**not preregistered**) | `scripts/final_test_report.py`, `src/evaluation/stats.py` | `90fd2ba` | **not before the test run:** committed 6 min 18 s after the stage-2 test run started (`6bb65f5`, 21:55:45 → 22:02:03 on 2026-10-28). The repository cannot show when the results were first read. See [`results/provenance/stage2_launch.md`](../results/provenance/stage2_launch.md) |
| Observed-days diagnostic (post hoc) | `results/posthoc_observed/PREREGISTRATION.md` | `c954d71` | its numbers were computed (on the already-used test set) |
| 4.1 Rolling origin | `results/rolling/PREREGISTRATION.md` | `0d36807` | training |
| 4.2 Chronos | `results/chronos/PREREGISTRATION.md` | `9122d6d` | prediction and scoring |
| 4.3 ACLED | `results/acled/PREREGISTRATION.md` | `205af44` | training; the lag was selected on validation before any test prediction existed |
| 4.4 Regime dummies | `results/regime/PREREGISTRATION.md` | `205af44` | training |
| 4.5 Transfer | `results/transfer/PREREGISTRATION.md` | `205af44` | training |
| 4.6 Conformal intervals | `results/conformal/PREREGISTRATION.md` | `8874076` | scoring |
| 4.7 FX target | `results/fx_target/PREREGISTRATION.md` | `2fcd848` | training |

Each report script refuses to run unless its preregistration is committed and unchanged, and it embeds that file's commit in its output. The commit hashes in this table refer to the original development history, which is not published; each report records its preregistration commit (for example `results/acled/acled_table.md`, first paragraph).

---

## 4. Baselines

Persistence (last value repeated), drift, ARIMA and ETS on log price (configuration chosen on validation), and for Sudan a VECM. Test MAE:

| track | h = 1 | h = 7 | h = 30 |
|---|---:|---:|---:|
| Sudan persistence (SDG/g) | 88.00 | 297.56 | 890.35 |
| Global persistence (USD/oz) | 13.72 | 27.01 | 51.80 |

None of the 21 statistical-baseline configurations beats persistence. The closest is global h = 7 ETS, at MASE 0.995, which is not significant. Baselines appear in every results table.

---

## 5. Results

### 5.1 Phase 3: scale fix, five models, three seeds (main split)

Values are mean ± std over seeds 1–3, from [`results/final/test_table.md`](../results/final/test_table.md). The "best" column is the configuration with the lowest seed-mean MASE among the validation-selected configurations and the selected ensemble.

| track, h | MinMax scaling MASE, 5 models | best selected configuration | its MAE | its MASE | DM vs persistence (abs / sq) |
|---|---|---|---:|---:|---|
| Sudan, 1 | 31.633–259.302 | TFT, log-return | 122.47 ± 13.16 | 1.392 | worse / no significant difference |
| Sudan, 7 | 13.544–76.286 | ensemble (selected) | 334.10 ± 13.51 | 1.123 | worse / no significant difference |
| Sudan, 30 | 3.652–25.963 | LSTM, RevIN | 1178.45 ± 46.83 | 1.324 | worse / no significant difference |
| Global, 1 | 1.643–38.245 | ensemble (selected) | 13.69 ± 0.03 | 0.998 | no significant difference |
| Global, 7 | 1.995–18.302 | ensemble (selected) | 27.18 ± 0.31 | 1.006 | no significant difference |
| Global, 30 | 1.747–12.876 | ensemble (selected) | 50.04 ± 0.87 | 0.966 | no significant difference |

- **The scale fix largely removes the failure.** Errors shrink from MinMax's 1.643–259.302× persistence to 0.966–1.392× for the best configuration of each setting. Not every selected configuration recovers: PatchTST with RevIN at Sudan h = 1 is still at MASE 12.761. The fix does not produce skill.
- **No configuration beats persistence.** 72 configurations (60 model × condition cells plus 12 ensembles), and none beats persistence in all three seeds. On Sudan, every selected configuration has a significantly higher absolute loss than persistence in all three seeds (before Holm; at h = 30 not after Holm).
- **Ensemble members** are mostly TFT and LSTM, chosen on validation ([`results/summary/ensembles.md`](../results/summary/ensembles.md)).

### 5.2 Headline across every experiment

From [`results/summary/headline.md`](../results/summary/headline.md), test windows, before multiple-testing correction:

| experiment | seeds | configurations | beat persistence (abs / sq) | lowest seed-mean MASE |
|---|---:|---:|---|---:|
| Phase 3 main split | 3 | 72 | 0 / 0 | 0.966 |
| statistical baselines | 1 | 21 | 0 / 0 | 0.995 |
| rolling origin, 5 folds pooled | 1 | 36 | 0 / 0 | 0.998 |
| Chronos zero-shot (mini, small) | 3 | 12 | 0 / 0 | 1.031 |
| FX-rate target, price rebuilt | 3 | 18 | 0 / 0 | 1.145 |
| ACLED conflict features, main split | 3 | 18 | 0 / 0 | 1.115 |
| ACLED conflict features, rolling | 1 | 6 | 0 / 0 | 1.088 |
| regime dummies, with and without | 3 | 36 | 0 / 0 | 1.032 |
| global → Sudan transfer and scratch | 3 | 12 | 0 / 0 | 1.085 |

---

## 6. Ablations and further experiments

Each experiment compares like with like, under the Phase 3 conditions of its models and a verdict rule committed in advance ([§3.4](#34-preregistration-and-test-use)).

- **4.1 Rolling origin** ([`results/rolling/rolling_table.md`](../results/rolling/rolling_table.md)).
  - Setup: five expanding folds, every model under its selected condition, plus baselines. One seed.
  - Result: pooled over folds, none of the 36 neural/ensemble configurations beats persistence.
  - Sudan h = 30 fold 1: ARIMA/ETS/VECM reach MASE 3.84 / 3.84 / 3.89, because a validation block that rose +50.3% made them pick trend models, then the test block fell 3.5%. This is not a bug. Without fold 1, their pooled MASE is 1.001 / 1.007 / 1.006 ([`results/rolling/fold1_sudan_h30_check.md`](../results/rolling/fold1_sudan_h30_check.md), post hoc).
- **4.2 Chronos zero-shot** ([`results/chronos/chronos_table.md`](../results/chronos/chronos_table.md)).
  - Setup: `chronos-t5-mini` (20,456,192 parameters) and `chronos-t5-small` (46,154,240), on the same windows, median of 20 samples, three sampling seeds.
  - Result: none of the 12 configurations beats persistence. The best is Sudan h = 30 mini, at MASE 1.031. The uncalibrated sample quantiles under-cover in every setting ([`results/conformal/intervals.md`](../results/conformal/intervals.md)).
- **4.3 ACLED conflict features** ([`results/acled/acled_table.md`](../results/acled/acled_table.md), [`results/acled/crosscheck.md`](../results/acled/crosscheck.md)).
  - Data: monthly political-violence events and fatalities for Sudan from the weekly file. Weeks that span two months are split pro rata by days. The series is cross-checked against ACLED's country-month file: r = 0.9958, totals 6218 vs 6217. Lags are 1, 2 and 3 completed months, with no look-ahead.
  - Main split: TFT and LSTM, 3 seeds. The only verdict is that conflict features hurt LSTM h = 7 at lag 3, and it does not survive Holm. All 18 cells show no consistent difference after Holm.
  - Rolling check (lag 2, chosen on validation, one seed): they hurt TFT h = 1 and help LSTM h = 1 and h = 7 after Holm. The LSTM h = 1 gain appears in only 2 of 5 folds. This rolling check is not 3-seed confirmed.
- **4.4 Regime dummies** ([`results/regime/regime_table.md`](../results/regime/regime_table.md)).
  - Design: with vs without, on rolling folds 4–5, where post-coup rows are in training, and on the main split.
  - After Holm, the dummies help nowhere and hurt 4 cells on absolute loss: fold 4 TFT h = 1, main-split TFT h = 1, LSTM h = 1 and LSTM h = 7.
  - Power is low: only 21 (fold 4) and 50 (fold 5) real, not forward-filled, `regime_2` = 1 training rows. On the main split `regime_2` is 0 in training and validation and switches on only at test. A null result on folds 4–5 is low-power, not evidence of no effect.
  - TFT's variable-selection weights (validation windows, descriptive, not attention) rank `regime_2` 3 / 3 / 1 of 33 on fold 4 at horizon h = 30. It is ranked lower elsewhere: on the main split 11/9/16, 26/10/10 and 11/14/5 (h = 1 / 7 / 30). It is first in one seed only.
- **4.5 Global → Sudan transfer** ([`results/transfer/transfer_table.md`](../results/transfer/transfer_table.md)).
  - Setup: pretrain on the global track using the 18 inputs both tracks share, with log-return targets, then fine-tune on Sudan. Compared with Sudan from scratch on the same inputs. GRU and PatchTST, 3 seeds.
  - Result: 0 of 6 cells shows a consistent difference.
  - The two models get different kinds of transfer. At h = 7, "freeze the first layer with weights" freezes 56,832 of 164,615 GRU weights (34.5%; 34.2–34.6% across horizons), which is partial freezing. For PatchTST it freezes only 512 of 49,223 (1.0%; 0.4–1.8%), which is effectively full fine-tuning from a global initialisation.
  - Pretraining used 4,187 global windows at h = 7, against 1,769 in Phase 3, because the shared inputs leave out the macro series that set Phase 3's later start date.
- **4.6 Prediction intervals** ([`results/conformal/intervals.md`](../results/conformal/intervals.md)).
  - Split conformal, calibrated per step on validation residuals, scored on test.
  - Coverage is close to nominal for some selected configurations and under-covers for others, most often at 80%. The widths are close to persistence's own conformal interval (width ratio 0.85–1.09 for the selected TFT, LSTM and ensemble).
- **4.7 FX rate as the target** ([`results/fx_target/fx_table.md`](../results/fx_target/fx_table.md)).
  - Setup: forecast SDG/USD and rebuild the price as P_t × forecast FX / FX_t (the price at the origin scaled by the forecast change in FX), compared against forecasting the price directly.
  - Result: 0 of 18 cells better, 2 worse; 0 beat persistence, 16 worse.
- **Observed days only (post hoc)** ([`results/posthoc_observed/observed_days.md`](../results/posthoc_observed/observed_days.md)).
  - Setup: scored only on target days that are real observations.
  - Result: 11/18 selected configurations are still worse than persistence, with median MASE 1.263. Preregistered decision: the hypothesis that forward-filling explains the gap is partly supported.

---

## 7. Interpretation

- **The MinMax failure is a scaling artefact.** MinMax scaling fitted on the training range cannot represent test prices 6.4–9.9× higher. Scale-free targets remove most of the error.
- **What remains is the difficulty of the series.** At these horizons, under this protocol, neither the deep models, nor Chronos, nor the extensions add information beyond the last observed price.
- **Sudan.** The price moves almost one-for-one with the official SDG/USD rate (R² of log price on log exchange rate = 0.9976), so forecasting it largely means forecasting the official exchange rate through a currency collapse and a coup. There are 223 or fewer validation windows from one 7-month period. In that setting the Sudan h = 30 reversal is expected: a validation advantage over persistence did not carry over to test.
- **Interpretation, not a tested result: official rate changes are policy decisions.** Because the series is an official reference price that moves almost one-for-one with the official SDG/USD rate, forecasting it largely means forecasting changes in the official rate, which are discrete policy decisions rather than market moves. This is consistent with persistence being hard to beat.
  - It is only partly consistent with the many zero-change days. Between consecutive quotes the official rate is unchanged in only 78 of 1424 steps (median absolute step 0.63%). Most zero-change days in the daily series therefore come from forward-filling days without a quote, not from an unchanged rate.
  - The largest moves come as single steps between quotes: 289.06 → 357.93 (2021-03-10 → 2021-03-22) and 454.62 → 567.80 (2022-03-07 → 2022-03-09). Across the 404-day gap the rate went from 79.06 to 231.56, and the data do not show when inside the gap that happened.
  - Evidence: [`results/summary/facts.md`](../results/summary/facts.md).
- **Global.** Gold at daily-to-monthly horizons behaves close to a random walk here. Every selected global configuration is within MASE 0.966–1.783 of persistence, and none differs significantly in its favour.
- **Directional accuracy.** The low raw directional scores on Sudan mostly reflect forward-filled zero changes, which a continuous forecast can never match. On non-zero changes, models are near 50% (the "DA non-zero" columns, with their ceiling, in every table).

---

## 8. Future work

- **An unplanned observation about the 18 shared inputs.** In the transfer experiment, the fine-tuned Sudan GRU and PatchTST models on the 18 shared inputs had lower seed-mean test MAE than the Phase 3 models with all 33 inputs in all 6 cells. The ratio of the two MAEs is 0.302–0.896 ([`results/transfer/transfer_table.md`](../results/transfer/transfer_table.md)). This was **not preregistered** and is **confounded**: the input set, the target condition and the comparison runs all differ, and none of these models beats persistence. It is a lead for a planned feature-ablation study, not a result.
- **Official vs parallel-market exchange rate.** The dataset uses the official rate. Comparing it with a parallel-market rate, and the gold price implied by each, would show how far the official series lags the market and whether a market rate is more forecastable. Parallel-market data are not in this repository.
- **Further directions.** Forecast distributions and calibration under regime change. Evaluation on series that are not forward-filled and have longer post-break histories. Data that reflect the exchange-rate regime directly.
