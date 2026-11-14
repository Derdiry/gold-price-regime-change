"""
Chronos wrapper (v1 code). NOT used for any reported result: the v2 Chronos runs
(chronos-t5-mini and chronos-t5-small, zero-shot only) use scripts/run_chronos.py.
No fine-tuned Chronos model was trained or evaluated.

Chronos is Amazon's pretrained time-series foundation model (chronos-forecasting package).

The wrapper was written to make Chronos compatible with:
  - The project's train/predict interface (same as LSTM/GRU/TFT)
  - MLflow pyfunc registration (so the serving platform can serve it)
  - Walk-forward validation (predict_fn callable)

Chronos is PyTorch-based (HuggingFace). It is the only non-TF model in this project.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Chronos pipeline wrapper
# ─────────────────────────────────────────────────────────────────────────────

class ChronosWrapper:
    """
    Wraps the Chronos HuggingFace pipeline with a sklearn-style interface
    so it plugs into WalkForwardValidator and the MLflow pyfunc system.

    Usage:
        wrapper = ChronosWrapper.from_pretrained("amazon/chronos-t5-mini")
        y_pred = wrapper.predict(X)   # X: (batch, seq_len, n_features)

    Note: Chronos is univariate — it only uses the target channel (index 0).
    All other feature channels are ignored. This is the baseline behaviour;
    multivariate conditioning is not used.
    """

    def __init__(self, pipeline, target_channel: int = 0, horizon: int = 1):
        self.pipeline = pipeline
        self.target_channel = target_channel
        self.horizon = horizon

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "amazon/chronos-t5-mini",
        device: str = "cpu",
        horizon: int = 1,
        target_channel: int = 0,
    ) -> "ChronosWrapper":
        """
        Load a Chronos model from HuggingFace Hub (v2 used amazon/chronos-t5-mini, 20,456,192
        parameters, and amazon/chronos-t5-small, 46,154,240; results/summary/facts.md).
        """
        try:
            from chronos import ChronosPipeline
            import torch
        except ImportError:
            raise ImportError(
                "pip install chronos-forecasting torch\n"
                "Chronos requires PyTorch — it runs separately from the TF models."
            )

        device_map = "cpu" if device == "cpu" else "cuda" if "cuda" in device else device
        logger.info(f"Loading Chronos from {model_id} on {device_map}")

        pipeline = ChronosPipeline.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype="bfloat16" if device_map != "cpu" else "float32",
        )
        return cls(pipeline, target_channel=target_channel, horizon=horizon)

    def predict(self, X: np.ndarray, horizon: Optional[int] = None) -> np.ndarray:
        """
        Run zero-shot Chronos inference.

        Args:
            X:       (batch, seq_len, n_features) — uses target_channel only
            horizon: override default horizon

        Returns:
            np.ndarray of shape (batch, horizon) — median forecast
        """
        import torch

        h = horizon or self.horizon
        # Extract target channel: (batch, seq_len)
        context = X[:, :, self.target_channel]

        results = []
        for i in range(len(context)):
            ctx_tensor = torch.tensor(context[i], dtype=torch.float32).unsqueeze(0)
            # Chronos returns (n_samples, batch, horizon) quantile forecasts
            forecast = self.pipeline.predict(
                context=ctx_tensor,
                prediction_length=h,
                num_samples=20,
            )
            # Use median (quantile 0.5) as point forecast
            median = np.quantile(forecast[0].numpy(), 0.5, axis=0)
            results.append(median)

        return np.array(results)  # (batch, horizon)

    def predict_with_quantiles(
        self,
        X: np.ndarray,
        quantiles: list = [0.1, 0.5, 0.9],
        horizon: Optional[int] = None,
    ) -> dict:
        """
        Run Chronos with quantile outputs for uncertainty estimation.

        Returns dict:
          {0.1: (batch, horizon), 0.5: (batch, horizon), 0.9: (batch, horizon)}
        """
        import torch

        h = horizon or self.horizon
        context = X[:, :, self.target_channel]

        all_quantiles = {q: [] for q in quantiles}

        for i in range(len(context)):
            ctx_tensor = torch.tensor(context[i], dtype=torch.float32).unsqueeze(0)
            forecast = self.pipeline.predict(
                context=ctx_tensor,
                prediction_length=h,
                num_samples=100,
            )
            samples = forecast[0].numpy()  # (n_samples, horizon)
            for q in quantiles:
                all_quantiles[q].append(np.quantile(samples, q, axis=0))

        return {q: np.array(v) for q, v in all_quantiles.items()}

    def fine_tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_epochs: int = 10,
        learning_rate: float = 1e-5,
        batch_size: int = 16,
    ) -> None:
        """
        Fine-tune Chronos on gold price training data.

        Uses HuggingFace Trainer for fine-tuning. Modifies pipeline in-place.
        Only the target channel (univariate) is used for fine-tuning.

        Note: Fine-tuning Chronos properly requires the chronos training scripts.
        This is a lightweight version using the pipeline's underlying model.
        """
        try:
            import torch
            from torch.optim import AdamW
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("torch required for Chronos fine-tuning")

        logger.info(
            f"Fine-tuning Chronos: {len(X_train)} samples, "
            f"{n_epochs} epochs, lr={learning_rate}"
        )

        context = X_train[:, :, self.target_channel]
        targets = y_train  # (N, horizon)

        ctx_tensor = torch.tensor(context, dtype=torch.float32)
        tgt_tensor = torch.tensor(targets, dtype=torch.float32)

        dataset = TensorDataset(ctx_tensor, tgt_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = self.pipeline.model
        optimizer = AdamW(model.parameters(), lr=learning_rate)

        model.train()
        for epoch in range(n_epochs):
            total_loss = 0.0
            for batch_ctx, batch_tgt in loader:
                optimizer.zero_grad()

                # Use the model's own loss (NLL over quantile tokens)
                outputs = model(
                    context=batch_ctx,
                    target=batch_tgt,
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            logger.info(f"  Epoch {epoch + 1}/{n_epochs} — loss: {avg_loss:.4f}")

        model.eval()
        logger.info("Chronos fine-tuning complete")

    def save(self, path: str) -> None:
        """Save the fine-tuned pipeline to disk."""
        Path(path).mkdir(parents=True, exist_ok=True)
        self.pipeline.model.save_pretrained(path)
        self.pipeline.tokenizer.save_pretrained(path)
        meta = {"target_channel": self.target_channel, "horizon": self.horizon}
        with open(os.path.join(path, "wrapper_meta.pkl"), "wb") as f:
            pickle.dump(meta, f)
        logger.info(f"Saved Chronos to {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "ChronosWrapper":
        """Load a saved fine-tuned Chronos pipeline."""
        try:
            from chronos import ChronosPipeline
        except ImportError:
            raise ImportError("pip install chronos-forecasting")

        pipeline = ChronosPipeline.from_pretrained(path, device_map=device)
        with open(os.path.join(path, "wrapper_meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        return cls(pipeline, **meta)


# ─────────────────────────────────────────────────────────────────────────────
# MLflow pyfunc wrapper for serving
# ─────────────────────────────────────────────────────────────────────────────

class ChronosPyfunc:
    """
    Minimal mlflow.pyfunc-compatible wrapper for Chronos.
    Bakes target_channel and horizon into the artifact so the serving
    platform receives raw arrays and returns predictions.

    Registered separately from GoldPricePyfunc because Chronos uses
    its own serialisation (HuggingFace save_pretrained).
    """

    def __init__(self, wrapper: ChronosWrapper, target_scaler=None):
        self.wrapper = wrapper
        self.target_scaler = target_scaler

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Run inference. Handles inverse-scaling if target_scaler is set.

        X: (batch, seq_len, n_features)
        Returns: (batch, horizon)
        """
        y_pred = self.wrapper.predict(X)

        if self.target_scaler is not None:
            shape = y_pred.shape
            y_pred = self.target_scaler.inverse_transform(
                y_pred.reshape(-1, 1)
            ).reshape(shape)

        return y_pred


# ─────────────────────────────────────────────────────────────────────────────
# Convenience builder
# ─────────────────────────────────────────────────────────────────────────────

def build_chronos_from_config(cfg: dict, horizon: int) -> ChronosWrapper:
    """Build and return a ChronosWrapper using model_config.yaml."""
    cc = cfg["models"]["chronos"]
    return ChronosWrapper.from_pretrained(
        model_id=cc["model_id"],
        horizon=horizon,
    )
