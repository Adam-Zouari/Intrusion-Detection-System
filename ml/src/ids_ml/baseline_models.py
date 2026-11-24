"""Scikit-learn baseline model adapters."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .data import (
    LABEL_ORDER,
    RANDOM_STATE,
    balanced_sample_weights,
    decode_labels,
    encode_labels,
    make_inner_split,
    transformed_feature_count,
)
from .epoch_logging import (
    EpochFitResult,
    EpochReporter,
    fixed_label_macro_f1,
    inner_split_context,
    log_epoch_artifacts,
)
from .experiment_specs import ExperimentSpec
from .preprocessing import (
    OneHotPipelineAdapterBase,
    make_one_hot_pipeline,
)
from .screening import AdapterMetadata, FitDetails


def build_baseline_pipeline(
    spec: ExperimentSpec,
    feature_sets: dict[str, list[str]],
    smoke: bool = False,
) -> Pipeline:
    weighted = spec.weighting_mode == "balanced"
    scale_numeric = spec.model_key in {"sgd", "mlp"}
    if spec.model_key == "dummy":
        classifier = DummyClassifier(strategy="most_frequent")
    elif spec.model_key == "sgd":
        classifier = SGDClassifier(
            loss="log_loss",
            class_weight="balanced" if weighted else None,
            max_iter=3 if smoke else 1_000,
            tol=None if smoke else 1e-3,
            n_jobs=1 if smoke else -1,
            random_state=RANDOM_STATE,
        )
    elif spec.model_key == "decision_tree":
        classifier = DecisionTreeClassifier(
            max_depth=4 if smoke else 20,
            min_samples_leaf=5,
            class_weight="balanced" if weighted else None,
            random_state=RANDOM_STATE,
        )
    elif spec.model_key == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=3 if smoke else 100,
            max_depth=4 if smoke else 20,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample" if weighted else None,
            n_jobs=1 if smoke else -1,
            random_state=RANDOM_STATE,
        )
    elif spec.model_key == "hist_gradient_boosting":
        classifier = HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_iter=3 if smoke else 100,
            max_leaf_nodes=7 if smoke else 31,
            l2_regularization=1.0,
            class_weight="balanced" if weighted else None,
            early_stopping=not smoke,
            random_state=RANDOM_STATE,
        )
    elif spec.model_key == "mlp":
        classifier = MLPClassifier(
            hidden_layer_sizes=(16,) if smoke else (128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=128 if smoke else 2_048,
            learning_rate_init=1e-3,
            max_iter=1 if smoke else 50,
            early_stopping=False,
            n_iter_no_change=5,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown baseline model: {spec.model_key}")

    return make_one_hot_pipeline(
        feature_sets[spec.feature_set],
        StandardScaler() if scale_numeric else "passthrough",
        np.float64,
        classifier,
    )


def _weighting_mechanism(spec: ExperimentSpec) -> str:
    if spec.model_key == "mlp" and spec.weighting_mode == "balanced":
        return "balanced_sample_weight"
    if spec.weighting_mode == "balanced":
        return "classifier_class_weight"
    return "none"


def _fit_mlp_phase(
    template: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    balanced: bool,
    epochs: int,
    stopping: tuple[pd.DataFrame, pd.Series] | None = None,
) -> tuple[Pipeline, pd.DataFrame, int]:
    """Fit one fresh pipeline, advancing the classifier's RNG across epochs."""
    pipeline = clone(template)
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["classifier"]
    model.set_params(random_state=np.random.RandomState(RANDOM_STATE))
    X_training = preprocessor.fit_transform(X)
    y_training = encode_labels(y)
    weights = balanced_sample_weights(y) if balanced else None
    reporter = EpochReporter("selection" if stopping is not None else "refit")
    options = {} if weights is None else {"sample_weight": weights}
    if stopping is not None:
        X_stopping = preprocessor.transform(stopping[0])
        y_stopping = encode_labels(stopping[1])
    classes = np.arange(len(LABEL_ORDER))
    best_epoch, best_score = 0, -np.inf
    patience_anchor = -np.inf
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        start = perf_counter()
        model.partial_fit(
            X_training,
            y_training,
            classes=classes if epoch == 1 else None,
            **options,
        )
        if stopping is None:
            reporter.record(epoch, start, model.loss_)
            continue
        score = fixed_label_macro_f1(y_stopping, model.predict(X_stopping))
        reporter.record(epoch, start, model.loss_, stopping_macro_f1=score)
        # Keep the earliest absolute best score. Require at least tol
        # improvement over the patience anchor to reset the stopping clock.
        if score > best_score:
            best_epoch, best_score = epoch, score
        if score > patience_anchor and score >= patience_anchor + model.tol:
            patience_anchor = score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= model.n_iter_no_change:
            break
    return pipeline, reporter.history(), best_epoch if stopping is not None else epochs


class BaselineAdapter(OneHotPipelineAdapterBase):
    """Expose one scikit-learn configuration through the shared runner contract."""

    def __init__(
        self,
        spec: ExperimentSpec,
        feature_sets: dict[str, list[str]],
        smoke: bool = False,
    ) -> None:
        if spec.protocol_mode != "one_hot":
            raise ValueError("Scikit-learn baselines require one-hot Protocol input.")
        self.spec = spec
        expected_transformed_count = transformed_feature_count(
            len(feature_sets[spec.feature_set]), spec.protocol_mode
        )
        super().__init__(
            build_baseline_pipeline(spec, feature_sets, smoke=smoke),
            expected_transformed_count,
        )
        self.fit_result: EpochFitResult | None = None
        self.inner_context: dict[str, object] = {}
        target_encoding = (
            "fixed_label_index" if spec.model_key == "mlp" else "original_string_labels"
        )
        self.metadata = AdapterMetadata(
            model_library="scikit-learn",
            model_library_version=sklearn.__version__,
            weighting_mechanism=_weighting_mechanism(spec),
            target_encoding=target_encoding,
            protocol_encoding=spec.protocol_mode,
            numeric_preprocessing=(
                "standard_scaler" if spec.model_key in {"sgd", "mlp"} else "passthrough"
            ),
            numeric_dtype="float64",
        )

    def parameters(self) -> dict[str, object]:
        parameters = super().parameters()
        if self.spec.model_key == "mlp":
            parameters["early_stopping_metric"] = "inner_validation_macro_f1"
            parameters["classifier__hidden_layer_sizes"] = str(
                self.pipeline.named_steps["classifier"].hidden_layer_sizes
            )
        return parameters

    def fit(self, X: pd.DataFrame, y: pd.Series) -> FitDetails:
        if self.spec.model_key == "mlp":
            return self._fit_mlp(X, y)
        self.pipeline.fit(X, y)
        self.inspect_fitted_preprocessor()
        return FitDetails()

    def _fit_mlp(self, X: pd.DataFrame, y: pd.Series) -> FitDetails:
        split = make_inner_split(y)
        balanced = self.spec.weighting_mode == "balanced"
        selection_start = perf_counter()
        provisional, selection_history, best_epoch = _fit_mlp_phase(
            self.pipeline,
            X.iloc[split.training_positions],
            y.iloc[split.training_positions],
            balanced=balanced,
            epochs=self.pipeline.named_steps["classifier"].max_iter,
            stopping=(X.iloc[split.stopping_positions], y.iloc[split.stopping_positions]),
        )
        selection_time = perf_counter() - selection_start
        del provisional
        refit_start = perf_counter()
        self.pipeline, final_history, _ = _fit_mlp_phase(
            self.pipeline, X, y, balanced=balanced, epochs=best_epoch
        )
        refit_time = perf_counter() - refit_start
        self.inspect_fitted_preprocessor()
        self.inner_context = inner_split_context(y, split)
        self.fit_result = EpochFitResult(
            best_epoch, selection_time, refit_time, selection_history, final_history
        )
        return FitDetails(metrics=self.fit_result.metrics())

    def fitted_context(self) -> dict[str, object]:
        return {**super().fitted_context(), **self.inner_context}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = np.asarray(self.pipeline.predict(X))
        if self.spec.model_key == "mlp":
            return decode_labels(predictions.astype(np.int64))
        return predictions

    def log_diagnostics(self, X: pd.DataFrame, y: pd.Series) -> None:
        if self.fit_result is not None:
            log_epoch_artifacts(self.fit_result)


def build_adapter(
    spec: ExperimentSpec,
    feature_sets: dict[str, list[str]],
    smoke: bool = False,
) -> BaselineAdapter:
    return BaselineAdapter(spec, feature_sets, smoke=smoke)
