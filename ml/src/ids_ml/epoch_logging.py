"""Live and saved epoch diagnostics shared by neural training workflows."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .data import LABEL_ORDER, InnerSplit
from .evaluation import log_line_figure, log_table_artifact


def fixed_label_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Score encoded labels on the shared, unweighted class order."""
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=range(len(LABEL_ORDER)),
            average="macro",
            zero_division=0,
        )
    )


class EpochReporter:
    """Record completed epochs separately for selection and fixed refitting."""

    def __init__(self, phase: Literal["selection", "refit"]) -> None:
        self.phase = phase
        self.records: list[dict[str, int | float]] = []

    def record(
        self,
        epoch: int,
        started_at: float,
        training_loss: float,
        *,
        stopping_macro_f1: float | None = None,
        learning_rate: float | None = None,
    ) -> None:
        elapsed = perf_counter() - started_at
        if (stopping_macro_f1 is not None) != (self.phase == "selection"):
            raise ValueError("Only selection epochs must have an inner-validation score.")
        record: dict[str, int | float] = {
            "epoch": epoch,
            "training_loss": float(training_loss),
            "epoch_duration_seconds": elapsed,
        }
        prefix = "" if self.phase == "selection" else "refit_"
        metrics = {
            f"{prefix}training_loss": float(training_loss),
            f"{prefix}epoch_duration_seconds": elapsed,
        }
        score_text = ""
        if stopping_macro_f1 is not None:
            record["stopping_macro_f1"] = float(stopping_macro_f1)
            metrics["inner_validation_macro_f1"] = float(stopping_macro_f1)
            score_text = f" | inner macro F1 {stopping_macro_f1:.6f}"
        if learning_rate is not None:
            record["lr"] = float(learning_rate)
        self.records.append(record)
        # In a real run, persist each epoch before proceeding to the next.
        # Smoke fits and direct estimator use must not implicitly create runs.
        if mlflow.active_run() is not None:
            mlflow.log_metrics(metrics, step=epoch, synchronous=True)
            print(
                f"{self.phase.capitalize()} epoch {epoch:2d} | "
                f"loss {training_loss:.6f}{score_text} | {elapsed:.2f} s",
                flush=True,
            )

    def history(self) -> pd.DataFrame:
        return pd.DataFrame(self.records).set_index("epoch")


@dataclass
class EpochFitResult:
    selected_epochs: int
    epoch_selection_time_seconds: float
    final_refit_time_seconds: float
    selection_history: pd.DataFrame
    final_history: pd.DataFrame

    def metrics(self) -> dict[str, float]:
        scores = self.selection_history["stopping_macro_f1"]
        return {
            "selected_epochs": float(self.selected_epochs),
            "best_epoch": float(self.selected_epochs),
            "epochs_trained": float(len(self.selection_history)),
            "best_inner_validation_macro_f1": float(scores.max()),
            "selected_inner_validation_macro_f1": float(scores.loc[self.selected_epochs]),
            "epoch_selection_time_seconds": self.epoch_selection_time_seconds,
            "final_refit_time_seconds": self.final_refit_time_seconds,
        }


def inner_split_context(labels: pd.Series, split: InnerSplit) -> dict[str, object]:
    def support(positions):
        counts = labels.iloc[positions].value_counts()
        return {label: int(counts.get(label, 0)) for label in LABEL_ORDER}

    return {
        "inner_training_rows": len(split.training_positions),
        "inner_stopping_rows": len(split.stopping_positions),
        "inner_training_fingerprint": split.training_fingerprint,
        "inner_stopping_fingerprint": split.stopping_fingerprint,
        "inner_training_support": support(split.training_positions),
        "inner_stopping_support": support(split.stopping_positions),
    }


def log_epoch_artifacts(result: EpochFitResult) -> None:
    log_table_artifact(
        result.selection_history, "epoch_selection_history.csv", "training"
    )
    log_line_figure(
        result.selection_history,
        ["training_loss", "stopping_macro_f1"],
        "epoch_selection_curves.png",
        "Inner stopping-set epoch selection",
        "training",
    )
    log_table_artifact(result.final_history, "final_refit_history.csv", "training")
