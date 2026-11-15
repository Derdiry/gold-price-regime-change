# Implementation notes

Design decisions, data handling and run provenance, each with its evidence. The README "Limitations" section, `docs/DETAILED_REPORT.md` and `docs/CLAIMS.md` draw on this file.

## Design decisions

| # | Decision | Why | Evidence |
|---|---|---|---|
| D1 | Scale-free targets (log-return and RevIN) are evaluated next to MinMax scaling, and each model's condition is chosen on validation only | Sudan test prices are 6.4–9.9× the training maximum, outside the range a MinMax scaler fitted on training rows can represent | `results/summary/facts.md`, `results/final/test_table.md`, `src/data/transforms.py`, `tests/test_transforms.py` |
| D2 | Model selection and ensemble members and weights use validation predictions only; `--resume` refuses a saved record without matching validation metrics | Test data must not influence any choice | `train.py` (`load_reusable_record`), `src/models/ensemble.py` (`fit_validation_ensemble`), `tests/test_ensemble.py`, `tests/test_resume.py` |
| D3 | Windows with any undefined feature are dropped, never back-filled; all 29 global features are kept, so global training starts in 2010-08 (1,775 windows) | Back-filling would copy later prices into earlier inputs. Global training therefore covers 2010-08 → 2017-09 only | `src/data/processors.py` (`row_validity`), `results/data/split_summary.md` |
| D4 | Both tracks are rebuilt from the raw files on every run; the data manifests record the SHA-256 of every input file | Processed data must be reproducible from the committed code | `src/data/pipeline.py`, `data/processed/v2/data_manifest_{global,sudan}.json` |
| D5 | The 404-day gap (2019-11-25 → 2021-01-02) is flagged `in_long_gap`, and every window whose context or target touches it is dropped; `is_ffilled` flags all 1,483 forward-filled days | Forward-filling the gap would create 403 flat days across the end of training and the start of validation | `results/data/split_summary.md`, `tests/test_windows.py` |
| D6 | The decimal-shift outlier (Open = 117,472 on 2016-12-03) is restored to 1,174.72 at load time; the raw file is unchanged | It would inflate the `open` feature's range about 100× | `src/data/quality.py` (`fix_decimal_shift_outliers`), `tests/test_quality.py` |
| D7 | Sudan dates with several quotes (635 dates, 1,103 extra rows of 2,528): the **first** row of the day in file order is kept | A fixed, deterministic rule. "Keep last" changes no baseline metric (largest change 0.00%), because all 635 multi-row dates fall in 2015-02-26 → 2019-11-25, in training rows only; its effect on the neural models was not tested | `src/data/quality.py` (`collapse_duplicate_dates`), `results/baselines/sensitivity_duplicate_last/comparison.md` |
| D8 | Directional accuracy differences each horizon step across consecutive origins, and is also reported on non-zero true changes with its attainable ceiling | Flattening `(windows, horizon)` arrays would compare values across window boundaries; forward-filled zero changes would dominate the raw score | `src/evaluation/metrics.py`, `tests/test_metrics.py` |
| D9 | Chronological 70/15/15 split by rows | One fixed split shared by every model and baseline | `src/data/processors.py` (`chronological_split`) |
| D10 | Global raw data: a 2026-04-04 Yahoo/FRED snapshot (not redistributed), hashed in the manifest | A fresh download changes history (revisions, adjusted prices) | `data/processed/v2/data_manifest_global.json` |
| D11 | World Bank and regional-price sources are off in every reported run; ACLED is used only in the Phase 4.3 experiment | Only the experiment designed for conflict data uses it | `configs/model_config.yaml`, `results/acled/PREREGISTRATION.md` |
| D12 | TFT batch size 8 was used in every run after a memory check before Phase 3 (logs not retained); hyperparameters were otherwise not tuned | Memory on an 8 GB CPU machine | `configs/model_config.yaml` (`batch_size_overrides`) |
| D13 | The effective batch size is logged per run, and done flags are written only after the record, predictions and MLflow run are saved | Records must state what actually ran, and `--resume` must never skip an incomplete run | `results/runs/*.json`, `train.py` |

## Data provenance

- `data/raw/new_Data.csv` (the ICCNC 2024 paper's dataset): official reference prices for Sudanese 21-karat gold (2015–2022), obtained by the author; not redistributable; available on request for research purposes. The Sudan series tracks the official SDG/USD exchange rate closely (R² of log price on log exchange rate = 0.9976, a fit of two trending series), so forecasting it largely means forecasting the official exchange rate (OLS on the 1425 observed rows, `results/summary/facts.md`); the `local USD` column (`fx_rate`) is the official rate.
- The full Sudan file is not included in this repository. `data/raw/SUDAN_DATA.md` describes it, and `data/raw/new_Data_SYNTHETIC_sample.csv` (written by `scripts/make_synthetic_sample.py`) shows its column schema with synthetic values only.

## Known limitations

- Global training window is 2010-08 → 2017-09 (D3).
- Sudan validation has 223/217/194 windows (h = 1/7/30), all after the gap (2021-03 → 2021-10); most of them start their context within 200 days of the gap, so their 200-day indicators (e.g. `sma_200`) still include forward-filled gap days.
- The global raw snapshot is not redistributable (Yahoo Finance / FRED terms); `--refetch-global` reproduces the pipeline but not byte-identical data.

## Baselines

- **No statistical baseline beats persistence materially.** Validation-selected ARIMA/ETS on log price are within 0.5% of persistence MAE on both tracks; the Sudan VECM is 0.3–6% worse; drift is worse everywhere. Evidence: `results/baselines/summary.md`.
- **Reading the raw directional score for baselines:** persistence scores 66.3% on Sudan h = 1 against a 21.1% "ceiling" because its predicted change between consecutive origins equals the previous true change, which is often zero (credited as a match). Use the `_nonzero` and `da_origin_nonzero` columns.

## Phase 3 stage 2 provenance

- Stage 2 (`scripts/phase3_stage2.py`) was launched at 2026-10-28 21:55:45 from a **clean tree at commit `6bb65f5`**. The tracked code/config diff against that commit is empty (`results/provenance/stage2_launch_code.patch`, 0 bytes), and the training-code fingerprint of the working tree equals that of `6bb65f5`.
- The test-comparison script (`90fd2ba`) was committed at 22:02:03, 6 min 18 s after launch; the Phase 3 main comparison is therefore not preregistered.
- Stage-2 records carry one of the commits from `6bb65f5` up to the last commit made before stage 2 finished. None of them changes training code. Details: `results/provenance/stage2_launch.md`.
- Commit hashes refer to the original development history, which is not published.

## Phase 4 provenance

- Chronos (4.2), conformal intervals with Chronos quantiles (4.6), FX target (4.7) and rolling origin (4.1) ran from 2026-10-29 10:00 to 2026-10-30 03:50 with no commits in between; all 225 run records of these runs (75 in `results/fx_target/runs/`, 150 seed-1 records in `results/rolling/fold*/runs/`) carry the same commit and `git_dirty=False`. The other 24 records in `results/rolling/fold*/runs/` are the later seed-2 and seed-3 runs of the regime experiment.
- Rolling origin uses **one seed per fold** (stated in `results/rolling/PREREGISTRATION.md`). Its folds overlap the Phase 3 validation and test periods, so they are not untouched data.
