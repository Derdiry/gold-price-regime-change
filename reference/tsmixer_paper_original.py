# coding=utf-8
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
# Modified by Mohamed Elbushra (hybrid model of Wan & Elbushra, ICCNC 2024):
# res_block adds an LSTM(64), a Conv1D(64, kernel 1) and a MaxPooling1D(2, same) layer to the
# feature-mixing part; commented-out attention/LSTM/Conv1D experiments; build_model docstring.
# This file is therefore not the original Google Research TSMixer.

"""Implementation of TSMixer."""

import tensorflow as tf
from tensorflow.keras import layers


def res_block(inputs, norm_type, activation, dropout, ff_dim):
  """Residual block of TSMixer."""

  norm = (
      layers.LayerNormalization
      if norm_type == 'L'
      else layers.BatchNormalization
  )

  # Temporal Linear
  x = norm(axis=[-2, -1])(inputs)
  x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
  ## Add attention mechanism
  #attention = tf.keras.layers.Attention()([x, x])
  #x_attention = tf.concat([x, attention], axis=-1)
  x = layers.Dense(x.shape[-1], activation=activation)(x)
  x = layers.Dense(x.shape[-1], activation=activation)(x)
  x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Input Length, Channel]
  x = layers.Dropout(dropout)(x)
  res = x + inputs

  # Feature Linear
  x = norm(axis=[-2, -1])(res)

  # Add LSTM layer
  x = tf.keras.layers.LSTM(units=64, return_sequences=True)(x)
  # Add attention mechanism
  #attention = tf.keras.layers.Attention()([x, x])
  #x_attention = tf.concat([x, attention], axis=-1)
  # Add 1D Convolutional layer
  x = tf.keras.layers.Conv1D(64, kernel_size=1, strides=1)(x)

  # Add Max Pooling layer
  x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='same')(x)
  x = layers.Dense(x.shape[-1], activation=activation)(x)
  x = layers.Dense(ff_dim, activation=activation)(
      x
  )  # [Batch, Input Length, FF_Dim]
  x = layers.Dropout(dropout)(x)
  x = layers.Dense(inputs.shape[-1])(x)  # [Batch, Input Length, Channel]
  x = layers.Dropout(dropout)(x)
  return x + res


def build_model(

        input_shape,
        pred_len,
        norm_type,
        activation,
        n_block,
        dropout,
        ff_dim,
        target_slice,
):
  """Build TSMixer model with adjusted LSTM layer."""
  inputs = tf.keras.Input(shape=input_shape)
  x = inputs  # [Batch, Input Length, Channel]
  for _ in range(n_block):
    x = res_block(x, norm_type, activation, dropout, ff_dim)

  if target_slice:
    x = x[:, :, target_slice]

  x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
  # Add attention mechanism
  #attention = tf.keras.layers.Attention()([x, x])

  # Concatenate attention output with the original input
  #x = tf.concat([x, attention], axis=-1)
  # Adjust LSTM units based on input and output sequence lengths
  #lstm_units = min(input_shape[1], pred_len) // 2  # Adjust based on input and output sequence lengths
  # Add LSTM layer
  #x = tf.keras.layers.LSTM(units=lstm_units, return_sequences=True)(x)
  # Add 1D Convolutional layer
  #x = tf.keras.layers.Conv1D(64, kernel_size=1, strides=1)(x)
  # Add 1D MaxPooling layer
  #x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='same')(x)

  x = layers.Dense(pred_len)(x)  # [Batch, Channel, Output Length]
  outputs = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Output Length, Channel]

  return tf.keras.Model(inputs, outputs)

