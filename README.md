# Gold price forecasting under regime change

Can deep forecasters beat a naive "no change" forecast (persistence) for gold prices when a currency collapses? This repository re-evaluates and extends Wan & Elbushra (ICCNC 2024) on two series: Sudanese gold (SDG/g, 2015–2022, during which the pound lost 98.4% of its USD value) and international gold (USD/oz). It tests LSTM, GRU, a hybrid TSMixer, PatchTST, TFT, validation-selected ensembles, zero-shot Chronos and the usual extensions. The MinMax setup evaluated here is this repository's own earlier pipeline; the published paper's protocol and results were not re-run here.

**Headline: 0 of 231 forecast evaluations (overlapping across experiments) beat persistence** (Diebold–Mariano with HLN correction and Newey–West variance, p < 0.05 with the same sign in every seed, absolute or squared loss, before multiple-testing correction; [`results/summary/headline.md`](results/summary/headline.md)). The forecast files behind the tables are not distributed; among them, predictions for 9 of the 231 evaluations (6 rolling-origin and 3 FX-route ensembles) are not stored, so those 9 cannot be re-scored from forecast files; their verdicts come from the saved result tables. **At Sudan h = 30, two configurations that beat persistence on validation in all three seeds were worse on test** (validation 2021-03 → 2021-10; test from 2021-10-12): TFT (RevIN) and the validation-selected ensemble had lower validation MAE than persistence (797.77) in all three seeds (784.9 / 769.6 / 777.0 and 772.7 / 759.1 / 757.2), but higher test MAE, with MASE 1.522 and 1.374 ([`results/final/test_table.md`](results/final/test_table.md)).

## Results at a glance

Best validation-selected configuration per track and horizon (lowest seed-mean test MASE over seeds 1–3; MASE = MAE / MAE(persistence) on the same test windows, below 1 is better than persistence). Source: [`results/final/test_table.md`](results/final/test_table.md). The "best" column is chosen on test MASE and is descriptive only; every verdict used validation-only selection.

| track | h | best configuration | test MASE | significantly better than persistence? | significantly worse (absolute loss, all 3 seeds)? |
|---|---:|---|---:|---|---|
| Sudan (SDG/g) | 1 | TFT, log-return | 1.392 | no | yes, also after Holm |
| Sudan (SDG/g) | 7 | ensemble (selected) | 1.123 | no | yes, also after Holm |
| Sudan (SDG/g) | 30 | LSTM, RevIN | 1.324 | no | yes before Holm, not after |
| Global (USD/oz) | 1 | ensemble (selected) | 0.998 | no | no |
| Global (USD/oz) | 7 | ensemble (selected) | 1.006 | no | no |
| Global (USD/oz) | 30 | ensemble (selected) | 0.966 | no | no |

On squared loss none of these six differs significantly from persistence. The 231 configurations span the main split (72), statistical baselines (21), rolling origin (36), Chronos (12), an FX-rate target (18), ACLED conflict features (18 + 6), regime dummies (36) and global → Sudan transfer (12). Configurations overlap between experiments, so 231 counts evaluations, not distinct models.

## What this repository shows

**1. A scaling failure and its fix.** Sudan test targets are 6.4–9.9× the training maximum (3270 SDG/g). With inputs and targets scaled by a MinMax scaler fitted on the training rows, test MAE at Sudan h = 1 was 31.633–259.302 times persistence's. Two scale-free conditions, log-return targets and per-window normalisation (RevIN), each chosen per model on validation only, bring test MASE to 1.123–12.761 on Sudan and 0.966–1.783 on the global track. The fix **largely** removes the failure (best configuration per setting 0.966–1.392), but not for every model: PatchTST with RevIN at Sudan h = 1 is still at 12.761. It does not produce skill. Converting forecasts back to prices never reads the target window ([`tests/test_transforms.py`](tests/test_transforms.py)).

**2. Why persistence is hard to beat here.** The Sudan series tracks the official SDG/USD exchange rate closely (R² of log price on log exchange rate = 0.9976, a fit of two trending series), so forecasting it largely means forecasting the official exchange rate (OLS on the 1425 observed rows). The target is forward-filled on 1483 of 2908 days ([`results/summary/facts.md`](results/summary/facts.md)).

**3. The evaluation protocol.** Chronological 70/15/15 split with a 60-day input window and horizons 1, 7 and 30. Three seeds on the main split; five rolling-origin folds with one seed. Everything is scored on the same test windows in price units against persistence, drift, ARIMA, ETS and (Sudan) a VECM. Diebold–Mariano tests on absolute and squared loss use the HLN correction and Newey–West variance (lag h − 1). A verdict needs p < 0.05 with the same sign in all seeds, and Holm adjustment is applied per family ([`src/evaluation/stats.py`](src/evaluation/stats.py)). Model selection and ensemble weights use validation data only. The protocol scores each final configuration on test once; this cannot be verified from repository files.

**4. What was and was not preregistered.** Each Phase 4 experiment (rolling origin, Chronos, ACLED, regime dummies, transfer, conformal intervals, FX target) and the post-hoc observed-days diagnostic has a `PREREGISTRATION.md` in `results/` that was committed in the (unpublished) development history before its test predictions were scored; the commit hashes are recorded in each report. The FX-target preregistration had one section removed for this release (marked in the file). The **Phase 3 main comparison was not preregistered**: its selection rule was committed before any test prediction existed, but its test-comparison script was committed 6 min 18 s after the stage-2 test run had started ([`docs/DETAILED_REPORT.md` §3.4](docs/DETAILED_REPORT.md#34-preregistration-and-test-use)). The statistical baselines have no preregistration.

**5. A claim ledger.** Every number in this README traces to a generated file: [`docs/CLAIMS.md`](docs/CLAIMS.md) lists each claim with its result and evidence file, and [`docs/PROJECT_FACTS.md`](docs/PROJECT_FACTS.md) collects the verified facts. The full audit (data, protocol, every table, ablations, interpretation) is in [`docs/DETAILED_REPORT.md`](docs/DETAILED_REPORT.md), and the design decisions are in [`docs/implementation_notes.md`](docs/implementation_notes.md).

## Run it

### Fresh clone, no private data

Tested with Python 3.10 on a CPU-only machine.

```bash
git clone https://github.com/Derdiry/gold-price-regime-change.git
cd gold-price-regime-change
python3.10 -m venv ../gold-venv && source ../gold-venv/bin/activate   # keep the venv outside the repository
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

python -m pytest -q                      # 42 passed, 1 skipped (that test needs the private Sudan file)
python scripts/release_checks.py . --allow-git   # secrets, personal paths, restricted data, compile, ensembles, baselines, claims
python scripts/check_claims.py           # every claim's evidence file exists and contains its numbers
python scripts/make_synthetic_sample.py  # rewrites data/raw/new_Data_SYNTHETIC_sample.csv (20 synthetic rows)
jupyter nbconvert --to notebook --execute notebooks/02_results.ipynb notebooks/03_ablations.ipynb --output-dir /tmp   # figures from results/
```

The synthetic sample shows the format of the Sudan file only. It has 20 rows, fewer than the 60-day input window, so it cannot be used to train or evaluate models.

### Requires private or third-party data

These steps do not run from a fresh clone:

| step | needs |
|---|---|
| Sudan data build, `train.py --track sudan`, `scripts/report_splits.py`, `scripts/project_facts.py`, `notebooks/01_data.ipynb` | the private Sudan file `data/raw/new_Data.csv` (see [Data availability](#data-availability)) |
| `train.py --track global` and the global data build | Yahoo Finance and FRED downloads via `src/data/fetchers.py` (`FRED_API_KEY`). A fresh download is not byte-identical to the 2026-04-04 snapshot used here |
| `scripts/acled_crosscheck.py`, the ACLED experiment | the user's own ACLED files in `data/raw/sudan/` |
| `scripts/make_tables.py`, `scripts/compare_regenerated.py` | the private Sudan file, the Yahoo/FRED snapshot, the ACLED files and the saved forecasts in `logs/v2/` (several GB, not distributed). From a fresh clone `make_tables.py` stops at its first step |

Full pipeline, given the data (CPU; long runs are resumable with `--resume` and refuse to start below 3 GiB free memory):

```bash
python scripts/report_splits.py                # data build, splits, window counts
python scripts/run_baselines.py                # Phase 2
bash scripts/phase3_selection.sh               # Phase 3 stage 1 (validation only)
python scripts/select_transforms.py
python scripts/phase3_stage2.py                # Phase 3 stage 2 (seeds 1-3, test once)
python scripts/final_test_report.py
python scripts/phase4_rolling.py               # 4.1
python scripts/run_chronos.py predict && python scripts/run_chronos.py score   # 4.2
python scripts/acled_crosscheck.py             # 4.3 data
python scripts/phase4_regime_transfer_acled.py # 4.4, 4.5, 4.3
python scripts/conformal_intervals.py          # 4.6
python scripts/phase4_fx_target.py             # 4.7
python scripts/make_tables.py && python scripts/compare_regenerated.py
```

**Regeneration check.** With all of these inputs supplied, all 26 steps of `make_tables.py` run and every number in the regenerated tables equals the committed one. Three things differ from the committed files: provenance fields (commit, `git_dirty`); the preregistration commit hashes, which the report scripts take from the current git history and would replace with this repository's commit, because the original history is not published; and the "original-setup windows" column of `results/data/split_summary.md`, whose source files no longer exist. Regenerated tables should therefore not be committed as they are.

A single run: `python train.py --track sudan --horizon 7 --condition log_return --model tft --seeds 1`. Every run writes a record to `results/**/runs/` with metrics, parameter counts, the git commit and the full configuration. In this public copy, `results/acled/crosscheck.{md,json}` keep only aggregate statistics, and `scripts/acled_crosscheck.py` writes only those.

## Limitations

- **Scope.** One dataset with two series and one chronological split. The Sudan sample ends in December 2022, before the April 2023 war.
- **Single test period.** Each track has one test period: Sudan 2021-10-12 → 2022-12-22, global 2021-05-10 → 2024-12-30. The rolling-origin folds re-use parts of the validation and test periods, so they are not independent test periods.
- **Look-ahead in two global macro inputs.** `clean_macro` (`src/data/processors.py:48`) applies `interpolate(method="time")` and then `ffill().bfill()` to the monthly FRED series. The daily CPI and M2 values between two monthly observations therefore depend on the next month's value, and no publication lag is applied. This affects only these two global-track inputs. It was neither corrected nor measured. No global configuration beats persistence.
- **Small, unrepresentative Sudan validation set.** It has 223 / 217 / 194 windows, all after the 404-day gap and within a 7-month period (2021-03 → 2021-10), so condition and ensemble selection on Sudan is noisy.
- **Rolling origin.** One seed per fold. The folds overlap the Phase 3 validation and test periods, so they are not untouched data. Its verdicts are on pooled windows.
- **Regime test power.** Only 21 / 50 real post-coup training rows on folds 4 / 5, and 0 on the main split. Null results there are low-power, not evidence of no effect.
- **ACLED.** The rolling check uses one seed and is not 3-seed confirmed. The main-split effects do not survive Holm. ACLED data cannot be redistributed, so its numbers can only be reproduced with the user's own ACLED files.
- **Hyperparameters.** Hyperparameters were fixed in advance and not tuned (the TFT batch size of 8 was set for memory). A tuned model might do better. This study tests these configurations, not the best achievable ones.
- **Phase 3 was not preregistered.** Its selection rule was committed before any test prediction. Its test-comparison script was committed 6 min 18 s after the stage-2 test run began. The repository cannot show when the results were first read ([`docs/DETAILED_REPORT.md` §3.4](docs/DETAILED_REPORT.md#34-preregistration-and-test-use)).
- **Global data.** The raw data is a snapshot that cannot be redistributed. Refetching changes history, so global numbers reproduce exactly only from the snapshot.

## Data availability

| data | in this repository? |
|---|---|
| Sudan (`data/raw/new_Data.csv`, 2528 rows, 2015-01-06 → 2022-12-22): official reference prices for Sudanese 21-karat gold (2015–2022), obtained by the author; not redistributable; available on request for research purposes | **No.** [`data/raw/SUDAN_DATA.md`](data/raw/SUDAN_DATA.md) describes it, and [`data/raw/new_Data_SYNTHETIC_sample.csv`](data/raw/new_Data_SYNTHETIC_sample.csv) holds 20 **synthetic** rows in the same schema |
| Global: Yahoo Finance (gold futures, GLD, UUP, BNO) and FRED (DGS10, CPI, M2), 2026-04-04 snapshot | No (third-party terms); rebuild with `src/data/fetchers.py` |
| ACLED weekly *Aggregated data on Africa* and *political violence events by country-month-year* | No (ACLED terms); `results/acled/` holds aggregate statistics only |
| Results: run records, metrics, Diebold–Mariano tests, tables, preregistrations | Yes, under `results/` (aggregate statistics only; no per-day or per-window series) |

## Citation

If you use this code, please cite it with the metadata in [`CITATION.cff`](CITATION.cff) and the earlier paper it re-evaluates and extends:

```text
Wan Yan and Mohamed Elbushra, "Forecasting Sudan Gold Prices with a Hybrid Deep Learning Approach,"
2024 International Conference on Cloud and Network Computing (ICCNC), Jinhua, China, pp. 71–78, May 2024.
DOI: 10.1109/ICCNC63989.2024.00020
```

The paper's hybrid TSMixer code is [`reference/tsmixer_paper_original.py`](reference/tsmixer_paper_original.py). It is a modified copy of Google Research's TSMixer that adds LSTM, Conv1D and MaxPooling1D layers to each residual block. [`src/models/tsmixer.py`](src/models/tsmixer.py) is its Keras 3 port. It is not a reproduction of the original TSMixer.

## Licence

Code: Apache License 2.0 ([`LICENSE`](LICENSE)). Third-party attribution and modified files: [`NOTICE`](NOTICE). The Sudan data are not included and not covered by this licence.

## Repository layout

```
train.py                      training entry point
src/data/                     loading, quality fixes, features, splits/windows, transforms, folds, ACLED
src/models/                   LSTM, GRU, TSMixer (hybrid), PatchTST, TFT, Chronos wrapper
src/evaluation/               metrics, baselines, Diebold–Mariano / MASE / Holm, plots
src/training/, src/utils/     training loop, MLflow tracking, provenance, logging, resource checks
scripts/                      experiment orchestrators, report scripts, make_tables.py, release checks
results/                      run records, metrics, preregistrations, tables (all generated)
results/summary/              headline.md, facts.md, ensembles.md
tests/                        unit tests (leakage, windows, metrics, stats, folds, ACLED)
notebooks/                    02 and 03: figures from results/ (static outputs); 01 needs the private data (no outputs)
docs/DETAILED_REPORT.md       the full audit: data, protocol, all result tables, ablations, interpretation
docs/PROJECT_FACTS.md         verified facts with evidence paths
docs/implementation_notes.md  design decisions, data handling, run provenance
docs/CLAIMS.md                claims, results and evidence files
data/raw/                     SUDAN_DATA.md (description) and new_Data_SYNTHETIC_sample.csv (20 synthetic rows)
data/processed/               metric files of the original MinMax runs (used by scripts/check_audit_persistence.py); v2/ data manifests
reference/                    the earlier paper's hybrid TSMixer (modified Google Research code, Apache-2.0)
LICENSE, NOTICE, CITATION.cff Apache-2.0; third-party attribution; citation metadata
```
