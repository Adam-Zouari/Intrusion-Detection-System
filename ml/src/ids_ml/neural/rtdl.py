"""RTDL MLP, ResNet, and FT-Transformer classifiers."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
import torch
from rtdl_revisiting_models import FTTransformer, MLP, ResNet
from torch import nn
from torch.utils.data import DataLoader

from ..data import (
    EXPECTED_PROTOCOL_VALUES,
    InnerSplit,
    LABEL_ORDER,
    encode_labels,
    transformed_feature_count,
)
from ..epoch_logging import EpochFitResult, EpochReporter, fixed_label_macro_f1
from ..experiment_specs import ExperimentSpec
from .preprocessing import (
    EARLY_STOPPING_PATIENCE,
    INFERENCE_BATCH_SIZE,
    MAX_EPOCHS,
    NeuralPreprocessor,
    NeuralResourceOwner,
    balanced_loss_weights,
    make_loader,
    release_torch_memory,
    set_reproducible_seed,
    training_device,
)


class RTDLFlowClassifier(NeuralResourceOwner):
    def __init__(
        self,
        spec: ExperimentSpec,
        selected_features: list[str],
        smoke: bool = False,
    ) -> None:
        if spec.model_key not in {"mlp", "resnet", "ft_transformer"}:
            raise ValueError(f"Unsupported RTDL model: {spec.model_key}")
        self.spec = spec
        self.selected_features = list(selected_features)
        self.protocol_mode = spec.protocol_mode
        self.max_epochs = 1 if smoke else MAX_EPOCHS
        self.patience = 1 if smoke else EARLY_STOPPING_PATIENCE
        self.batch_size = 128 if smoke else (
            2_048 if spec.model_key == "ft_transformer" else 4_096
        )
        self.training_device = training_device()
        self.model: nn.Module | None = None
        self.preprocessor: NeuralPreprocessor | None = None
        self.fit_result: EpochFitResult | None = None

    @property
    def transformed_feature_count(self) -> int:
        return transformed_feature_count(
            len(self.selected_features), self.protocol_mode
        )

    def _model_config(self) -> tuple[type[nn.Module], dict[str, object]]:
        common = {"d_out": len(LABEL_ORDER), "n_blocks": 3}
        if self.spec.model_key == "mlp":
            return MLP, {
                **common,
                "d_in": self.transformed_feature_count,
                "d_block": 256,
                "dropout": 0.1,
            }
        if self.spec.model_key == "resnet":
            return ResNet, {
                **common,
                "d_in": self.transformed_feature_count,
                "d_block": 256,
                "d_hidden": None,
                "d_hidden_multiplier": 2.0,
                "dropout1": 0.15,
                "dropout2": 0.0,
            }
        return FTTransformer, {
            **common,
            "n_cont_features": len(self.selected_features) - 1,
            "cat_cardinalities": [len(EXPECTED_PROTOCOL_VALUES)],
            "d_block": 128,
            "attention_n_heads": 8,
            "attention_dropout": 0.2,
            "ffn_d_hidden": None,
            "ffn_d_hidden_multiplier": 4 / 3,
            "ffn_dropout": 0.1,
            "residual_dropout": 0.0,
        }

    def _build_model(self) -> nn.Module:
        architecture, parameters = self._model_config()
        return architecture(**parameters)

    def _optimizer_config(self) -> dict[str, float]:
        learning_rate = 1e-4 if self.spec.model_key == "ft_transformer" else 1e-3
        return {"lr": learning_rate, "weight_decay": 1e-5}

    def _optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        parameters = (
            model.make_parameter_groups()
            if isinstance(model, FTTransformer)
            else model.parameters()
        )
        return torch.optim.AdamW(parameters, **self._optimizer_config())

    def resolved_parameters(self) -> dict[str, object]:
        architecture, parameters = self._model_config()
        return {
            "architecture": architecture.__name__,
            **{f"architecture__{name}": value for name, value in parameters.items()},
            "optimizer": torch.optim.AdamW.__name__,
            **{f"optimizer__{name}": value for name, value in self._optimizer_config().items()},
            "inner_validation_batch_size": INFERENCE_BATCH_SIZE,
            "early_stopping_min_delta": 1e-6,
        }

    def _forward(
        self, model: nn.Module, batch: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        if self.protocol_mode == "embedding":
            continuous = batch[0].to(self.training_device, non_blocking=True)
            categorical = batch[1].to(self.training_device, non_blocking=True)
            return model(continuous, categorical)
        return model(batch[0].to(self.training_device, non_blocking=True))

    def _train_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        loader: DataLoader,
    ) -> float:
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch in loader:
            targets = batch[-1].to(self.training_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(self._forward(model, batch), targets)
            loss.backward()
            optimizer.step()
            rows = len(targets)
            total_loss += float(loss.detach().cpu()) * rows
            total_rows += rows
        return total_loss / total_rows

    def _predict_arrays(
        self,
        model: nn.Module,
        arrays: np.ndarray | tuple[np.ndarray, np.ndarray],
        device: torch.device,
    ) -> np.ndarray:
        original_device = self.training_device
        self.training_device = device
        model.eval()
        predictions: list[np.ndarray] = []
        loader = make_loader(arrays, None, INFERENCE_BATCH_SIZE, shuffle=False)
        with torch.inference_mode():
            for batch in loader:
                predictions.append(self._forward(model, batch).argmax(dim=1).cpu().numpy())
        self.training_device = original_device
        return np.concatenate(predictions).astype(np.int64, copy=False)

    def _fit_model_for_epochs(
        self,
        arrays: np.ndarray | tuple[np.ndarray, np.ndarray],
        targets: np.ndarray,
        epochs: int,
    ) -> tuple[nn.Module, pd.DataFrame]:
        set_reproducible_seed()
        model = self._build_model().to(self.training_device)
        optimizer = self._optimizer(model)
        weights = (
            balanced_loss_weights(targets, self.training_device)
            if self.spec.weighting_mode == "balanced"
            else None
        )
        criterion = nn.CrossEntropyLoss(weight=weights)
        loader = make_loader(arrays, targets, self.batch_size, shuffle=True)
        reporter = EpochReporter("refit")
        for epoch in range(1, epochs + 1):
            start = perf_counter()
            loss = self._train_epoch(model, optimizer, criterion, loader)
            reporter.record(epoch, start, loss)
        return model, reporter.history()

    def fit_with_inner_selection(
        self, X: pd.DataFrame, y: pd.Series, inner_split: InnerSplit
    ) -> EpochFitResult:
        y_encoded = encode_labels(y)
        selection_start = perf_counter()
        inner_preprocessor = NeuralPreprocessor(
            self.selected_features, self.protocol_mode
        ).fit(X.iloc[inner_split.training_positions])
        training_arrays = inner_preprocessor.transform(
            X.iloc[inner_split.training_positions]
        )
        stopping_arrays = inner_preprocessor.transform(
            X.iloc[inner_split.stopping_positions]
        )
        y_training = y_encoded[inner_split.training_positions]
        y_stopping = y_encoded[inner_split.stopping_positions]

        set_reproducible_seed()
        provisional_model = self._build_model().to(self.training_device)
        optimizer = self._optimizer(provisional_model)
        weights = (
            balanced_loss_weights(y_training, self.training_device)
            if self.spec.weighting_mode == "balanced"
            else None
        )
        criterion = nn.CrossEntropyLoss(weight=weights)
        loader = make_loader(training_arrays, y_training, self.batch_size, shuffle=True)
        reporter = EpochReporter("selection")
        best_macro_f1 = -np.inf
        best_epoch = 1
        epochs_without_improvement = 0
        for epoch in range(1, self.max_epochs + 1):
            start = perf_counter()
            training_loss = self._train_epoch(
                provisional_model, optimizer, criterion, loader
            )
            stopping_predictions = self._predict_arrays(
                provisional_model, stopping_arrays, self.training_device
            )
            stopping_macro_f1 = fixed_label_macro_f1(y_stopping, stopping_predictions)
            reporter.record(
                epoch, start, training_loss, stopping_macro_f1=stopping_macro_f1
            )
            if stopping_macro_f1 > best_macro_f1 + 1e-6:
                best_macro_f1 = stopping_macro_f1
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= self.patience:
                break
        selection_time = perf_counter() - selection_start
        selection_history = reporter.history()

        del provisional_model, optimizer, criterion, loader, training_arrays
        del stopping_arrays, y_training, y_stopping, inner_preprocessor
        release_torch_memory()

        final_training_start = perf_counter()
        self.preprocessor = NeuralPreprocessor(
            self.selected_features, self.protocol_mode
        ).fit(X)
        final_arrays = self.preprocessor.transform(X)
        self.model, final_history = self._fit_model_for_epochs(
            final_arrays, y_encoded, best_epoch
        )
        final_refit_time = perf_counter() - final_training_start
        self.model = self.model.to("cpu")
        self.training_device = torch.device("cpu")
        del final_arrays, y_encoded
        release_torch_memory()

        self.fit_result = EpochFitResult(
            selected_epochs=best_epoch,
            epoch_selection_time_seconds=float(selection_time),
            final_refit_time_seconds=float(final_refit_time),
            selection_history=selection_history,
            final_history=final_history,
        )
        return self.fit_result

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("The neural classifier is not fitted.")
        predictions = []
        for start in range(0, len(X), INFERENCE_BATCH_SIZE):
            arrays = self.preprocessor.transform(
                X.iloc[start : start + INFERENCE_BATCH_SIZE]
            )
            predictions.append(
                self._predict_arrays(self.model, arrays, torch.device("cpu"))
            )
        return np.concatenate(predictions).astype(np.int64, copy=False)
