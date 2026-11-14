# Phase 4.5: global → Sudan transfer. Preregistration

Written and committed before any pretraining, fine-tuning or from-scratch
model was trained. `scripts/transfer_report.py` refuses to run unless this
file is committed and unchanged, and copies it into its results file.

## Question

Does initialising a Sudan model from a model trained on the global gold
price improve Sudan forecasts over training the same model from scratch on
the same inputs?

## Design

- **Models.** GRU and PatchTST (the original plan). Horizons 1, 7 and 30.
  Seeds 1, 2 and 3.
- **Inputs.** 18 inputs shared by both tracks (`--feature-set shared`,
  `src/data/features.shared_feature_cols`), in this fixed order:
  - the track's own price (`gold_close` for global, `price_sdg` for Sudan);
  - then sma_20, sma_50, sma_200, ema_12, ema_26, rsi, macd, macd_signal,
    macd_hist, bb_upper, bb_lower, bb_width, bb_pct, return_1d, return_5d,
    return_20d, rolling_vol, each computed from that track's own price.
- **Condition: log_return for pretraining, fine-tuning and from scratch**
  (author decision). The two price levels differ by about 100×.
  - The log-return target is scale-free.
  - Each track's window scaler is fitted on that track's own training
    windows (`src/data/transforms.WindowScaler`).
- **Pretrain.** Global main split.
  - With the 18 inputs, training origins start on 2000-12-22: 4,187 windows
    at h = 7. That is more than the 29-input global runs, which start in
    2010-08, because the macro inputs that set that start are not used.
  - Validation and test windows are the Phase 3 global windows.
  - Standard training: Adam lr 1e-3, at most 100 epochs, early stopping
    (patience 10) on global validation MAE.
- **Fine-tune.** Sudan main split (`--transfer-from-global`). Each run starts
  from the pretrained model with the same architecture, horizon and seed.
  - The first layer that has weights is frozen: GRU `gru_1` (56,832 of
    164,615 weights), PatchTST `patch_embed` (512 of 49,223 weights).
  - `training.fine_tune` settings: Adam lr 1e-4, at most 50 epochs, early
    stopping (patience 7) on Sudan validation MAE.
- **From scratch.** Sudan main split, the same 18 inputs and log_return,
  random initialisation, standard training as for pretraining.
- **Test windows.** The Phase 3 Sudan test windows (437 / 431 / 408 at
  h = 1 / 7 / 30). The report checks them from the saved origins and
  targets.

## Metrics

- MAE, MAPE, MASE vs persistence (seed mean ± sd).
- **Paired DM test** of transfer (model) vs scratch (reference), same seed:
  HLN correction, Newey–West lag h − 1, two-sided, squared and absolute loss.
  DM* < 0 means transfer has lower loss.
- **Secondary:**
  - DM of transfer and of scratch vs persistence;
  - MAE ratio vs the Phase 3 full-input Sudan model of the same architecture
    under its selected condition (descriptive).

## Verdict rules

**Per cell (model, h) and loss:**
- **"transfer helps":** DM* < 0 and p < 0.05 in all 3 seeds;
- **"transfer hurts":** DM* > 0 and p < 0.05 in all 3 seeds;
- **"no consistent difference":** otherwise.

**Holm:** the same rule on Holm-adjusted p, over the 18 seed-level tests
(6 cells × 3 seeds) per loss.

**Summary:** the number of the 6 cells that help, hurt or show no consistent
difference, under absolute loss (primary) and squared loss.

**Vs persistence:** "beats persistence" or "worse" under the same
all-3-seeds rule, per configuration (transfer, scratch).
