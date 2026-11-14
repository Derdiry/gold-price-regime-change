"""
GRU model for gold price forecasting.

Architecture mirrors LSTM exactly (same depth, units, dropout, dense head)
so that any performance difference is attributable to the recurrent cell
type, not model capacity.

GRU uses fewer parameters than LSTM (no separate cell state) — often trains
faster and generalises better on smaller datasets like the Sudan track.
"""

from tensorflow import keras


def build_gru(
    input_shape: tuple,
    horizon: int,
    units: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    dense_units: int = 64,
    activation: str = "relu",
) -> keras.Model:
    """
    Build a stacked GRU model.

    Args:
        input_shape:  (sequence_length, n_features)
        horizon:      number of time steps to predict
        units:        GRU hidden units per layer
        num_layers:   number of stacked GRU layers
        dropout:      dropout rate applied after each GRU layer
        dense_units:  units in the dense bottleneck layer
        activation:   activation for dense bottleneck

    Returns:
        Uncompiled keras.Model
    """
    inputs = keras.Input(shape=input_shape, name="input")
    x = inputs

    for i in range(num_layers):
        return_sequences = (i < num_layers - 1)
        x = keras.layers.GRU(
            units=units,
            return_sequences=return_sequences,
            name=f"gru_{i + 1}",
        )(x)
        x = keras.layers.Dropout(dropout, name=f"dropout_{i + 1}")(x)

    x = keras.layers.Dense(dense_units, activation=activation, name="dense_bottleneck")(x)
    x = keras.layers.Dropout(dropout, name="dropout_dense")(x)
    outputs = keras.layers.Dense(horizon, activation="linear", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="gru")
    return model


def build_gru_from_config(cfg: dict, input_shape: tuple, horizon: int) -> keras.Model:
    """Build GRU using hyperparameters from model_config.yaml."""
    mc = cfg["models"]["gru"]
    return build_gru(
        input_shape=input_shape,
        horizon=horizon,
        units=mc["units"],
        num_layers=mc["num_layers"],
        dropout=mc["dropout"],
        dense_units=mc["dense_units"],
        activation=mc["activation"],
    )
