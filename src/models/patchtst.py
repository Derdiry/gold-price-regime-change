"""
PatchTST — implemented in TF/Keras.

Reference: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
           Nie et al., ICLR 2023. https://arxiv.org/abs/2211.14730

Key idea: instead of feeding individual time steps to a transformer (which struggles
with long sequences), divide the sequence into overlapping patches (like image tokens
in ViT) and feed each patch as a single token. This dramatically reduces sequence
length and preserves local temporal structure within each token.

Architecture:
  Input (B, T, C)
  → Channel-independent: process each channel separately, stack results
  → Patching: (T,) → (n_patches, patch_len) with stride
  → Patch embedding: Linear projection → (n_patches, d_model)
  → Positional encoding (learnable)
  → N × Transformer encoder blocks (MultiHeadAttention + FFN + LayerNorm)
  → Flatten → Dense(horizon)
  → Output (B, horizon)  [single target]

Channel-independent means each feature channel is processed separately and
the results are combined — this is the default PatchTST setting that gives
strong results on benchmarks.

Motivation for the Sudan track:
  - Patches as tokens: fewer tokens per channel, so attention has less to fit
  - Channel-independent weights are shared across features, which keeps the
    parameter count small (~68K for 29 features, h=7, with the config in
    configs/model_config.yaml)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ─────────────────────────────────────────────────────────────────────────────
# Patch Embedding
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbedding(layers.Layer):
    """
    Splits a 1D time series into overlapping patches and linearly projects each.

    Input:  (batch, time_steps)           — single channel
    Output: (batch, n_patches, d_model)
    """

    def __init__(self, patch_len: int, stride: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.projection = layers.Dense(d_model, use_bias=False, name="patch_projection")

    def call(self, x):
        # x: (batch, time_steps)
        # Extract patches via strided slicing
        patches = tf.signal.frame(x, frame_length=self.patch_len, frame_step=self.stride)
        # patches: (batch, n_patches, patch_len)
        return self.projection(patches)  # (batch, n_patches, d_model)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"patch_len": self.patch_len, "stride": self.stride, "d_model": self.d_model})
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Learnable Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────

class LearnablePositionalEncoding(layers.Layer):
    """
    Learnable positional embeddings added to patch embeddings.
    Unlike sinusoidal PE, learned PE adapts to the patch structure.
    """

    def __init__(self, max_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(1, max_len, d_model),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_embedding[:, :seq_len, :]


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Encoder Block
# ─────────────────────────────────────────────────────────────────────────────

class TransformerEncoderBlock(layers.Layer):
    """
    Standard pre-norm transformer encoder block:
      LayerNorm → MultiHeadAttention → residual
      LayerNorm → FFN (Dense → GELU → Dense) → residual
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(
            num_heads=n_heads,
            key_dim=d_model // n_heads,
            dropout=dropout,
        )
        self.ffn1 = layers.Dense(d_ff, activation="gelu")
        self.ffn2 = layers.Dense(d_model)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        # Self-attention with pre-norm
        normed = self.norm1(x)
        attn_out = self.attn(normed, normed, training=training)
        x = x + self.drop1(attn_out, training=training)

        # FFN with pre-norm
        normed = self.norm2(x)
        ffn_out = self.ffn2(self.ffn1(normed))
        x = x + self.drop2(ffn_out, training=training)

        return x

    def get_config(self):
        cfg = super().get_config()
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# PatchTST model
# ─────────────────────────────────────────────────────────────────────────────

class PatchTSTModel(keras.Model):
    """
    Memory-efficient PatchTST using the batch-dimension trick.

    Instead of building 29 separate layer stacks (one per channel) in the
    computation graph — which creates 29x intermediate tensors — we merge
    channels into the batch dimension, run ONE shared transformer, then
    reshape back. This is how the original paper implements CI mode.

    Forward pass:
      (B, T, C) → transpose → (B, C, T)
               → reshape  → (B*C, T)
               → shared encoder → (B*C, n_patches*d_model)
               → reshape  → (B, C * n_patches * d_model)
               → Dense(horizon) → (B, horizon)
    """

    def __init__(self, seq_len, n_features, horizon, patch_len, stride,
                 d_model, n_heads, num_encoder_layers, d_ff, dropout, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.n_features = n_features
        self.horizon = horizon
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_patches = (seq_len - patch_len) // stride + 1

        # Shared layers — used for ALL channels
        self.patch_embed = PatchEmbedding(patch_len, stride, d_model, name="patch_embed")
        self.pos_enc = LearnablePositionalEncoding(self.n_patches + 10, d_model, name="pos_enc")
        self.input_drop = layers.Dropout(dropout, name="input_drop")
        self.encoder_blocks = [
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout, name=f"encoder_{i}")
            for i in range(num_encoder_layers)
        ]
        self.final_norm = layers.LayerNormalization(epsilon=1e-6, name="final_norm")
        self.final_drop = layers.Dropout(dropout, name="final_drop")
        self.output_dense = layers.Dense(horizon, name="output")

    def call(self, inputs, training=False):
        B = tf.shape(inputs)[0]
        C = self.n_features

        # (B, T, C) → (B, C, T)
        x = tf.transpose(inputs, perm=[0, 2, 1])
        # (B, C, T) → (B*C, T)
        x = tf.reshape(x, [B * C, self.seq_len])

        # Shared encoder
        x = self.patch_embed(x)                        # (B*C, n_patches, d_model)
        x = self.pos_enc(x)
        x = self.input_drop(x, training=training)
        for block in self.encoder_blocks:
            x = block(x, training=training)

        # (B*C, n_patches, d_model) → (B*C, n_patches*d_model)
        x = tf.reshape(x, [B * C, self.n_patches * self.d_model])

        # (B*C, n_patches*d_model) → (B, C * n_patches * d_model)
        x = tf.reshape(x, [B, C * self.n_patches * self.d_model])

        x = self.final_norm(x)
        x = self.final_drop(x, training=training)
        return self.output_dense(x)                    # (B, horizon)

    def get_config(self):
        config = super().get_config()
        config.update({
            "seq_len": self.seq_len,
            "n_features": self.n_features,
            "horizon": self.horizon,
            "patch_len": self.patch_len,
            "stride": self.stride,
            "d_model": self.d_model,
            "n_heads": len(self.encoder_blocks[0].attn._query_dense.kernel.shape) if self.encoder_blocks else 2,
            "num_encoder_layers": len(self.encoder_blocks),
            "d_ff": self.encoder_blocks[0].ffn1.units if self.encoder_blocks else 64,
            "dropout": self.input_drop.rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):
        super().build(input_shape)


def build_patchtst(
    input_shape: tuple,
    horizon: int,
    patch_len: int = 16,
    stride: int = 8,
    d_model: int = 32,
    n_heads: int = 2,
    num_encoder_layers: int = 2,
    d_ff: int = 64,
    dropout: float = 0.1,
    activation: str = "gelu",
) -> keras.Model:
    """
    Build memory-efficient channel-independent PatchTST.

    Uses the batch-dimension trick: all channels share one transformer,
    processed via reshape rather than one layer stack per channel.
    """
    seq_len, n_features = input_shape
    return PatchTSTModel(
        seq_len=seq_len,
        n_features=n_features,
        horizon=horizon,
        patch_len=patch_len,
        stride=stride,
        d_model=d_model,
        n_heads=n_heads,
        num_encoder_layers=num_encoder_layers,
        d_ff=d_ff,
        dropout=dropout,
        name="patchtst",
    )


def build_patchtst_from_config(cfg: dict, input_shape: tuple, horizon: int) -> keras.Model:
    """Build PatchTST using hyperparameters from model_config.yaml."""
    mc = cfg["models"]["patchtst"]
    return build_patchtst(
        input_shape=input_shape,
        horizon=horizon,
        patch_len=mc["patch_len"],
        stride=mc["stride"],
        d_model=mc["d_model"],
        n_heads=mc["n_heads"],
        num_encoder_layers=mc["num_encoder_layers"],
        d_ff=mc["d_ff"],
        dropout=mc["dropout"],
        activation=mc["activation"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight variant for Sudan (smaller dataset)
# ─────────────────────────────────────────────────────────────────────────────

def build_patchtst_small(input_shape: tuple, horizon: int) -> keras.Model:
    """
    Smaller PatchTST variant optimised for the Sudan track (~2500 rows).
    Reduces d_model and layers to prevent overfitting on limited data.
    Use this for Sudan fine-tuning unless transfer learning from global model.
    """
    return build_patchtst(
        input_shape=input_shape,
        horizon=horizon,
        patch_len=8,
        stride=4,
        d_model=64,
        n_heads=4,
        num_encoder_layers=2,
        d_ff=128,
        dropout=0.15,
    )
