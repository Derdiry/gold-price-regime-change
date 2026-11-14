# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by Mohamed Elbushra: derived from reference/tsmixer_paper_original.py, itself a modified
# copy of Google Research's TSMixer. Changes: LSTM, Conv1D and MaxPooling1D layers added to the
# residual block (hybrid model of Wan & Elbushra, ICCNC 2024); ported to Keras 3; single-target
# (batch, horizon) output; config-based builder.

"""
Hybrid TSMixer (Wan & Elbushra, ICCNC 2024), refactored from the paper code
(reference/tsmixer_paper_original.py).

This is NOT a reproduction of the original TSMixer. Google Research's residual block is
modified by adding LSTM, Conv1D and MaxPooling1D layers to the feature-mixing part.

Architecture is the same as in reference/tsmixer_paper_original.py:
  - Residual block with:
      * Temporal Linear mixing (transpose → dense → dense → transpose)
      * LSTM (64 units, return_sequences=True)
      * Conv1D (64 filters, kernel=1)
      * MaxPooling1D (pool=2, same padding)
      * Feature Linear (Dense → ff_dim → Dense back to channels)
      * Dropout + residual connections
  - N stacked residual blocks
  - Optional target channel slicing
  - Transpose → Dense(pred_len) → transpose output head

Changes from reference/tsmixer_paper_original.py:
  - Wrapped in a class-style build function with config integration
  - Output reshaped to (batch, horizon) for single-target forecasting
  - Added build_tsmixer_from_config() for pipeline compatibility
  - No architecture changes relative to reference/tsmixer_paper_original.py (which itself
    differs from Google Research's TSMixer, see above)
"""

from tensorflow import keras
from tensorflow.keras import layers
import keras.ops as kops


# ─────────────────────────────────────────────────────────────────────────────
# Residual block (same as reference/tsmixer_paper_original.py)
# ─────────────────────────────────────────────────────────────────────────────

def res_block(inputs, norm_type: str, activation: str, dropout: float, ff_dim: int):
    """
    Hybrid TSMixer residual block (as in reference/tsmixer_paper_original.py).

    Flow:
      inputs
        → LayerNorm/BatchNorm
        → Temporal Linear (transpose → 2×Dense → transpose)
        → Dropout + residual (res)
        → LayerNorm/BatchNorm
        → LSTM(64, return_sequences=True)
        → Conv1D(64, kernel=1)
        → MaxPooling1D(pool=2, same)
        → Dense → Dense(ff_dim) → Dense(channels)
        → Dropout + residual
    """
    norm = (
        layers.LayerNormalization
        if norm_type == "L"
        else layers.BatchNormalization
    )

    # Temporal Linear
    x = norm(axis=[-2, -1])(inputs)
    x = kops.transpose(x, axes=[0, 2, 1])          # [B, Channel, Time]
    x = layers.Dense(x.shape[-1], activation=activation)(x)
    x = layers.Dense(x.shape[-1], activation=activation)(x)
    x = kops.transpose(x, axes=[0, 2, 1])          # [B, Time, Channel]
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feature Linear (with LSTM + CNN from paper)
    x = norm(axis=[-2, -1])(res)
    x = layers.LSTM(units=64, return_sequences=True)(x)
    x = layers.Conv1D(64, kernel_size=1, strides=1)(x)
    x = layers.MaxPooling1D(pool_size=2, strides=1, padding="same")(x)
    x = layers.Dense(x.shape[-1], activation=activation)(x)
    x = layers.Dense(ff_dim, activation=activation)(x)   # [B, Time, ff_dim]
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1])(x)                # [B, Time, Channel]
    x = layers.Dropout(dropout)(x)

    return x + res


# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_tsmixer(
    input_shape: tuple,
    horizon: int,
    n_block: int = 4,
    ff_dim: int = 64,
    dropout: float = 0.2,
    norm_type: str = "L",
    activation: str = "relu",
    target_slice=None,
) -> keras.Model:
    """
    Build the hybrid TSMixer model (architecture of reference/tsmixer_paper_original.py).

    Args:
        input_shape:  (sequence_length, n_features)
        horizon:      number of future steps to predict
        n_block:      number of residual blocks
        ff_dim:       feed-forward dimension in feature linear layer
        dropout:      dropout rate
        norm_type:    "L" = LayerNorm, "B" = BatchNorm
        activation:   activation function name
        target_slice: slice object to select target channels before output head.
                      Use slice(0, 1) for single-target (predicts only first channel).
                      None = use all channels.

    Returns:
        Uncompiled keras.Model with output shape (batch, horizon, n_out_channels)
        For single-target use, squeeze/index output[:, :, 0] after prediction.
    """
    inputs = keras.Input(shape=input_shape, name="input")
    x = inputs  # [B, Time, Channel]

    for _ in range(n_block):
        x = res_block(x, norm_type, activation, dropout, ff_dim)

    if target_slice is not None:
        x = x[:, :, target_slice]   # [B, Time, target_channels]

    x = kops.transpose(x, axes=[0, 2, 1])     # [B, Channel, Time]
    x = layers.Dense(horizon)(x)            # [B, Channel, horizon]
    outputs = kops.transpose(x, axes=[0, 2, 1])  # [B, horizon, Channel]

    model = keras.Model(inputs=inputs, outputs=outputs, name="tsmixer")
    return model


def build_tsmixer_single_target(
    input_shape: tuple,
    horizon: int,
    n_block: int = 4,
    ff_dim: int = 64,
    dropout: float = 0.2,
    norm_type: str = "L",
    activation: str = "relu",
) -> keras.Model:
    """
    TSMixer variant with a flat (batch, horizon) output for single-target use.

    Adds a squeeze layer after the standard output head so the model
    produces the same shape as LSTM/GRU (batch, horizon).
    Uses target_slice=slice(0, 1) to predict only the first channel.
    """
    base = build_tsmixer(
        input_shape=input_shape,
        horizon=horizon,
        n_block=n_block,
        ff_dim=ff_dim,
        dropout=dropout,
        norm_type=norm_type,
        activation=activation,
        target_slice=slice(0, 1),  # only predict target channel
    )

    # Squeeze channel dimension: (B, horizon, 1) → (B, horizon)
    inputs = base.input
    x = base.output
    outputs = layers.Reshape((horizon,), name="squeeze")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="tsmixer")
    return model


def build_tsmixer_from_config(cfg: dict, input_shape: tuple, horizon: int) -> keras.Model:
    """Build TSMixer using hyperparameters from model_config.yaml."""
    mc = cfg["models"]["tsmixer"]
    return build_tsmixer_single_target(
        input_shape=input_shape,
        horizon=horizon,
        n_block=mc["n_block"],
        ff_dim=mc["ff_dim"],
        dropout=mc["dropout"],
        norm_type=mc["norm_type"],
        activation=mc["activation"],
    )
