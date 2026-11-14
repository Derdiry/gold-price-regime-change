"""
LSTM model for gold price forecasting.

Architecture:
  - 2 stacked LSTM layers (128 hidden units each)
  - Dropout after each LSTM layer
  - Dense bottleneck (64 units, ReLU)
  - Output Dense (horizon units, linear)

Designed to match the GRU exactly in depth and capacity for a fair comparison.
"""

from tensorflow import keras


def build_lstm(
    input_shape: tuple,
    horizon: int,
    units: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    dense_units: int = 64,
    activation: str = "relu",
) -> keras.Model:
    """
    Build a stacked LSTM model.

    Args:
        input_shape:  (sequence_length, n_features)
        horizon:      number of time steps to predict
        units:        LSTM hidden units per layer
        num_layers:   number of stacked LSTM layers
        dropout:      dropout rate applied after each LSTM layer
        dense_units:  units in the dense bottleneck layer
        activation:   activation for dense bottleneck

    Returns:
        Uncompiled keras.Model
    """
    inputs = keras.Input(shape=input_shape, name="input")
    x = inputs

    for i in range(num_layers):
        return_sequences = (i < num_layers - 1)
        x = keras.layers.LSTM(
            units=units,
            return_sequences=return_sequences,
            name=f"lstm_{i + 1}",
        )(x)
        x = keras.layers.Dropout(dropout, name=f"dropout_{i + 1}")(x)

    x = keras.layers.Dense(dense_units, activation=activation, name="dense_bottleneck")(x)
    x = keras.layers.Dropout(dropout, name="dropout_dense")(x)
    outputs = keras.layers.Dense(horizon, activation="linear", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="lstm")
    return model


def build_lstm_from_config(cfg: dict, input_shape: tuple, horizon: int) -> keras.Model:
    """Build LSTM using hyperparameters from model_config.yaml."""
    mc = cfg["models"]["lstm"]
    return build_lstm(
        input_shape=input_shape,
        horizon=horizon,
        units=mc["units"],
        num_layers=mc["num_layers"],
        dropout=mc["dropout"],
        dense_units=mc["dense_units"],
        activation=mc["activation"],
    )
