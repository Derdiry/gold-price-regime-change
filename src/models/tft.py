"""
Temporal Fusion Transformer (TFT) — implemented in TF/Keras.

Reference: "Temporal Fusion Transformers for Interpretable Multi-horizon
            Time Series Forecasting", Lim et al., 2021.
            https://arxiv.org/abs/1912.09363

Its key outputs:
  - Variable Selection Networks (VSN): per-feature selection weights. The TFT feature
    rankings reported in results/ are these weights (averaged over validation windows),
    not attention weights.
  - Temporal self-attention: shows which past time steps influence predictions
  - Quantile forecasts: built-in uncertainty (optional)

Architecture:
  Input features (past values + known future covariates + static metadata)
  → Variable Selection Networks (separate for static, encoder, decoder)
  → Gated Residual Networks (GRN) for non-linear processing
  → LSTM encoder (processes past sequence)
  → LSTM decoder (generates future context)
  → Temporal Self-Attention over encoder+decoder outputs
  → Position-wise Feed-Forward
  → Quantile output heads

Simplified version here:
  - Past features only (no future known covariates) to match other models' input contract
  - Single quantile output (point forecast) — can extend to quantile later
  - Variable selection weights logged to MLflow for feature importance
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Gated Residual Network (GRN)
# ─────────────────────────────────────────────────────────────────────────────

@tf.keras.utils.register_keras_serializable(package="tft")
class GatedResidualNetwork(layers.Layer):
    """
    Core building block of TFT.

    GRN(x, c=None):
      η₁ = ELU(Dense(x) + Dense(c)) if context c is provided
      η₂ = Dense(η₁)
      gate = sigmoid(Dense(x))   [Gated Linear Unit]
      output = LayerNorm(x_proj + gate ⊙ η₂)

    Context c allows the network to condition on additional information
    (used in Variable Selection to condition on static features).
    """

    def __init__(self, units: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout
        self.dense1 = layers.Dense(units, activation="elu")
        self.dense2 = layers.Dense(units)
        self.gate = layers.Dense(units, activation="sigmoid")
        self.proj = layers.Dense(units)   # for residual when input dim ≠ units
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.drop = layers.Dropout(dropout)  # dropout_rate stored in self.dropout_rate

    def call(self, x, context=None, training=False):
        residual = self.proj(x)

        if context is not None:
            x = self.dense1(x) + layers.Dense(self.units)(context)
        else:
            x = self.dense1(x)

        x = self.dense2(x)
        x = self.drop(x, training=training)
        gate = self.gate(residual)
        x = gate * x
        return self.norm(residual + x)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units, "dropout": self.dropout_rate})
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Variable Selection Network (VSN)
# ─────────────────────────────────────────────────────────────────────────────

@tf.keras.utils.register_keras_serializable(package="tft")
class VariableSelectionNetwork(layers.Layer):
    """
    Learns soft weights over input features indicating their relevance.

    Input:  (batch, time, n_features) or (batch, n_features) for static
    Output: (batch, time, units), selection_weights (batch, time, n_features)

    The selection_weights are the feature importance scores stored in
    results/**/tft_importance/ (descriptive, validation windows).
    """

    def __init__(self, n_features: int, units: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.units = units
        self.dropout_rate = dropout
        self.grns = [GatedResidualNetwork(units, dropout, name=f"grn_var_{i}")
                     for i in range(n_features)]
        self.softmax_grn = GatedResidualNetwork(n_features, dropout, name="grn_softmax")
        self.softmax = layers.Activation("softmax")
        self.feature_projs = [layers.Dense(units, name=f"feat_proj_{i}")
                              for i in range(n_features)]

    def call(self, x, context=None, training=False):
        # x: (batch, time, n_features) or (batch, n_features)
        is_temporal = len(x.shape) == 3

        if is_temporal:
            # Process each feature independently
            processed = [
                self.grns[i](
                    self.feature_projs[i](x[:, :, i:i+1]),
                    context=context,
                    training=training,
                )
                for i in range(self.n_features)
            ]
            # Stack: (batch, time, n_features, units)
            stacked = tf.stack(processed, axis=2)

            # Variable selection weights
            flat = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1],
                                  self.n_features])
            weights = self.softmax_grn(flat, context=context, training=training)
            weights = self.softmax(weights)  # (batch, time, n_features)
            weights_expanded = tf.expand_dims(weights, axis=-1)

            # Weighted combination: (batch, time, units)
            output = tf.reduce_sum(stacked * weights_expanded, axis=2)
        else:
            processed = [
                self.grns[i](
                    self.feature_projs[i](x[:, i:i+1]),
                    context=context,
                    training=training,
                )
                for i in range(self.n_features)
            ]
            stacked = tf.stack(processed, axis=1)  # (batch, n_features, units)
            weights = self.softmax(
                self.softmax_grn(x, context=context, training=training)
            )  # (batch, n_features)
            weights_expanded = tf.expand_dims(weights, axis=-1)
            output = tf.reduce_sum(stacked * weights_expanded, axis=1)

        return output, weights

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_features": self.n_features, "units": self.units, "dropout": self.dropout_rate})
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Self-Attention with interpretable multi-head attention
# ─────────────────────────────────────────────────────────────────────────────

@tf.keras.utils.register_keras_serializable(package="tft")
class InterpretableMultiHeadAttention(layers.Layer):
    """
    TFT's interpretable attention: uses a shared value projection across heads
    so attention weights can be averaged across heads for a single interpretable
    attention score per time step.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model
        self.dropout_rate = dropout

        self.q_proj = layers.Dense(d_model)
        self.k_proj = layers.Dense(d_model)
        self.v_proj = layers.Dense(self.d_head)  # shared value projection
        self.out_proj = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout)
        self.scale = tf.math.sqrt(tf.cast(self.d_head, tf.float32))

    def call(self, x, training=False):
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)  # (B, T, d_head) — shared across heads

        # Split Q and K into heads
        Q = tf.reshape(Q, [B, T, self.n_heads, self.d_head])
        Q = tf.transpose(Q, [0, 2, 1, 3])  # (B, heads, T, d_head)
        K = tf.reshape(K, [B, T, self.n_heads, self.d_head])
        K = tf.transpose(K, [0, 2, 1, 3])

        # Attention scores
        attn = tf.matmul(Q, K, transpose_b=True) / self.scale  # (B, heads, T, T)
        # Causal mask: only attend to past
        causal_mask = tf.linalg.band_part(tf.ones((T, T)), -1, 0)
        attn = attn * causal_mask + (1 - causal_mask) * (-1e9)
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.dropout(attn, training=training)

        # Average attention across heads for interpretability
        attn_avg = tf.reduce_mean(attn, axis=1)  # (B, T, T)

        # Apply to shared value
        V_expanded = tf.expand_dims(V, 1)  # (B, 1, T, d_head)
        out = tf.matmul(attn, V_expanded)  # (B, heads, T, d_head)
        out = tf.reduce_mean(out, axis=1)  # (B, T, d_head)
        out = self.out_proj(out)           # (B, T, d_model)

        return out, attn_avg

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "n_heads": self.n_heads, "dropout": self.dropout_rate})
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# TFT model
# ─────────────────────────────────────────────────────────────────────────────

def build_tft(
    input_shape: tuple,
    horizon: int,
    hidden_size: int = 128,
    lstm_layers: int = 2,
    n_heads: int = 4,
    dropout: float = 0.1,
    ff_dim: int = 256,
) -> keras.Model:
    """
    Build Temporal Fusion Transformer in TF/Keras.

    This implementation uses past features only (matches other models' input
    contract: X shape = (batch, seq_len, n_features)).

    Feature importance is accessible by running the VSN sub-model on the
    same input and extracting the 'selection_weights' output.

    Args:
        input_shape:   (sequence_length, n_features)
        horizon:       forecast horizon
        hidden_size:   TFT hidden dimension (all GRN/LSTM units)
        lstm_layers:   LSTM encoder depth
        n_heads:       attention heads
        dropout:       dropout rate
        ff_dim:        position-wise FFN dimension

    Returns:
        Uncompiled keras.Model
    """
    seq_len, n_features = input_shape
    inputs = keras.Input(shape=input_shape, name="input")  # (B, T, C)

    # 1. Variable Selection Network
    vsn = VariableSelectionNetwork(n_features, hidden_size, dropout, name="vsn")
    selected, var_weights = vsn(inputs)  # (B, T, hidden_size), (B, T, n_features)

    # 2. LSTM encoder
    x = selected
    for i in range(lstm_layers):
        return_seq = True  # always return sequences for attention
        x = layers.LSTM(hidden_size, return_sequences=return_seq,
                        name=f"lstm_enc_{i}")(x)
        x = layers.Dropout(dropout, name=f"lstm_drop_{i}")(x)

    # 3. Add & Norm after LSTM
    x = GatedResidualNetwork(hidden_size, dropout, name="grn_post_lstm")(x)
    x = layers.LayerNormalization(epsilon=1e-6, name="norm_post_lstm")(selected + x)

    # 4. Temporal Self-Attention
    attn_layer = InterpretableMultiHeadAttention(
        hidden_size, n_heads, dropout, name="temporal_attn"
    )
    attn_out, attn_weights = attn_layer(x)
    attn_out = layers.Dropout(dropout, name="attn_drop")(attn_out)
    x = layers.LayerNormalization(epsilon=1e-6, name="norm_post_attn")(x + attn_out)

    # 5. Position-wise Feed-Forward (GRN)
    x = GatedResidualNetwork(hidden_size, dropout, name="grn_ffn")(x)
    x = layers.LayerNormalization(epsilon=1e-6, name="norm_ffn")(x)

    # 6. Output head — use last `horizon` time steps or flatten
    # Take the final hidden_size output at the last time step and project to horizon
    x_last = x[:, -1, :]                      # (B, hidden_size)
    x_last = layers.Dropout(dropout, name="output_drop")(x_last)
    outputs = layers.Dense(horizon, name="output")(x_last)  # (B, horizon)

    model = keras.Model(inputs=inputs, outputs=outputs, name="tft")
    return model


def build_tft_from_config(cfg: dict, input_shape: tuple, horizon: int) -> keras.Model:
    """Build TFT using hyperparameters from model_config.yaml."""
    mc = cfg["models"]["tft"]
    return build_tft(
        input_shape=input_shape,
        horizon=horizon,
        hidden_size=mc["hidden_size"],
        lstm_layers=mc["lstm_layers"],
        n_heads=mc["num_attention_heads"],
        dropout=mc["dropout"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_feature_importance(
    model: keras.Model,
    X: np.ndarray,
    feature_cols: list[str],
) -> dict:
    """
    Extract variable selection weights from the TFT model.

    Builds a sub-model that outputs the VSN weights and runs
    it on a batch of inputs. Returns mean weights across time and samples.

    Returns:
        {"feature_name": importance_score, ...}  (scores sum to ~1)
    """
    # Get the VSN layer
    vsn_layer = model.get_layer("vsn")

    # Build sub-model: input → VSN weights
    sub_inputs = model.input
    _, weights = vsn_layer(sub_inputs)
    vsn_model = keras.Model(inputs=sub_inputs, outputs=weights)

    # Run on data (use batches to avoid OOM)
    batch_size = 128
    all_weights = []
    for i in range(0, len(X), batch_size):
        batch = X[i : i + batch_size]
        w = vsn_model.predict(batch, verbose=0)
        all_weights.append(w)

    all_weights = np.concatenate(all_weights, axis=0)  # (N, T, n_features)
    mean_weights = all_weights.mean(axis=(0, 1))        # (n_features,)
    mean_weights = mean_weights / mean_weights.sum()    # normalise

    return dict(zip(feature_cols, mean_weights.tolist()))
