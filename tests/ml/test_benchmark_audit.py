"""Exercise the real screening loops and MLflow reader without the IDS dataset."""

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from ids_ml import baseline_models, screening_workflows as workflows
from ids_ml.data import DatasetContract, ExperimentData, LABEL_ORDER, index_fingerprint
from ids_ml.evaluation import TimingInputs
from ids_ml.reporting import build_screening_report


@pytest.fixture
def tracking_project(tmp_path: Path, monkeypatch):
    """Keep project discovery, SQLite, artifacts, and CSVs out of the real repo."""
    prior_uri = mlflow.get_tracking_uri()
    monkeypatch.setattr(mlflow.tracking.fluent, "_active_experiment_id", None)
    (tmp_path / "ml").mkdir()
    (tmp_path / "pyproject.toml").touch()
    monkeypatch.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        mlflow.end_run()
        mlflow.set_tracking_uri(prior_uri)


def test_four_mlp_runs_and_fresh_group_proof(tracking_project, monkeypatch):
    rng = np.random.default_rng(42)
    y = pd.Series(np.repeat(LABEL_ORDER, 30), index=rng.permutation(450))
    x = pd.DataFrame(
        rng.normal(size=(len(y), 70)), columns=[f"feature_{i}" for i in range(70)],
        index=y.index,
    )
    x.insert(0, "Protocol", rng.choice([0, 6, 17], len(y)))
    xv = x.iloc[:15].copy()
    xv.index += 1_000
    yv = pd.Series(LABEL_ORDER, index=xv.index)
    contract = DatasetContract(
        "synthetic", index_fingerprint(x.index), index_fingerprint(xv.index),
        "no-test-rows", "synthetic-timing",
    )
    data = ExperimentData(
        x, xv, pd.DataFrame(), y, yv, pd.Series(dtype=object), list(x.columns),
        {"all_71": list(x), "reduced_64": list(x)[:64]}, contract,
    )
    monkeypatch.setattr(workflows, "load_dataset_contract", lambda: contract)
    monkeypatch.setattr(workflows, "load_experiment_data", lambda **kwargs: data)
    timing_inputs = TimingInputs((xv.iloc[:1],), xv, contract.timing_fingerprint)
    monkeypatch.setattr(workflows, "make_timing_inputs", lambda _: timing_inputs)
    # Test the full tracked path but use a tiny network and one epoch.
    build_adapter = baseline_models.build_adapter
    monkeypatch.setattr(
        baseline_models, "build_adapter",
        lambda spec, feature_sets, **kw: build_adapter(spec, feature_sets, smoke=True),
    )
    assert workflows.baseline_main(["--models", "mlp", "--rerun"]) == 0
    report = build_screening_report(["baseline"], project_root=tracking_project)
    assert len(report.leaderboard) == 4
    assert set(zip(report.leaderboard.source_feature_count,
                   report.leaderboard.transformed_feature_count)) == {(71, 73), (64, 66)}
    matched = report.matching_runs
    assert matched["run_group_id"].nunique() == 1
    assert len(matched) == 4
    client = mlflow.MlflowClient()
    split_pairs = set()
    for run_id in report.leaderboard["run_id"]:
        run = client.get_run(run_id)
        assert run.info.status == "FINISHED"
        assert run.data.tags["mlflow.source.name"] == "ids-run-baselines"
        assert run.data.metrics["peak_training_ram_mib"] > 0
        assert "peak_training_vram_mib" not in run.data.metrics
        artifacts = {item.path for item in client.list_artifacts(run_id)}
        assert "environment.json" in artifacts
        context = client.download_artifacts(run_id, "evaluation/run_context.json")
        metadata = json.loads(Path(context).read_text(encoding="utf-8"))
        split_pairs.add((
            metadata["inner_training_fingerprint"], metadata["inner_stopping_fingerprint"]
        ))
        assert metadata["inner_training_rows"] + metadata["inner_stopping_rows"] == len(x)
    assert len(split_pairs) == 1
    assert report.leaderboard["peak_training_ram_mib"].gt(0).all()

    # A normal invocation skips these four, but a broken explicit rerun
    # must not pass based on the historical four successful records.
    assert workflows.baseline_main(["--models", "mlp"]) == 0
    monkeypatch.setattr(
        workflows, "run_validation_experiment", lambda *args, **kwargs: {"macro_f1": 0.0}
    )
    assert workflows.baseline_main(["--models", "mlp", "--rerun"]) != 0


@pytest.mark.parametrize("model", ["mlp", "tabnet"])
def test_neural_run_persists_both_epoch_phases(tracking_project, model):
    """Use native training loops; on CPU-only CI this still tests both phases."""
    import torch
    from ids_ml.experiment_specs import specs_for_round
    from ids_ml.neural.experiments import build_adapter
    from ids_ml.screening import run_validation_experiment, ValidationPartition
    from ids_ml.tracking import setup_mlflow_experiment

    rng = np.random.default_rng(81)
    y = pd.Series(np.repeat(LABEL_ORDER, 20))
    X = pd.DataFrame({"Protocol": rng.choice([0, 6, 17], len(y)), "Bytes": rng.random(len(y))})
    xv = X.sample(30, random_state=42).reset_index(drop=True)
    xv.index += 1000
    yv = pd.Series(LABEL_ORDER * 2, index=xv.index)
    spec = next(
        s for s in specs_for_round("neural")
        if (s.model_key, s.feature_set, s.weighting_mode) == (model, "all_71", "balanced")
    )
    data = ExperimentData(X, xv, pd.DataFrame(), y, yv, pd.Series(dtype=object),
                          list(X), {"all_71": list(X)}, DatasetContract("synthetic"))
    prior_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        setup_mlflow_experiment("synthetic-neural", tracking_project)
        result = run_validation_experiment(
            spec, data, None, lambda: build_adapter(spec, data.feature_sets, smoke=True),
            partition=ValidationPartition(X, y, xv, yv, 42, "smoke"),
        )
        client = mlflow.MlflowClient()
        run_id = result["run_id"]
        detail = client.get_run(run_id).data
        for metric in ("training_loss", "inner_validation_macro_f1", "epoch_duration_seconds",
                       "refit_training_loss", "refit_epoch_duration_seconds"):
            history = client.get_metric_history(run_id, metric)
            assert [v.step for v in history] == [1]
            assert np.isfinite(history[0].value)
        assert detail.metrics["peak_training_ram_mib"] > 0
        assert detail.tags["mlflow.source.name"] == "ids-run-neural"
        if torch.cuda.is_available():
            assert detail.tags["training_device"] == "cuda"
            assert detail.metrics["peak_training_vram_mib"] > 0
        assert detail.tags["inference_device"] == "cpu"
        assert set(p.path for p in client.list_artifacts(run_id, "training")) >= {
            "training/epoch_selection_history.csv", "training/final_refit_history.csv"
        }
    finally:
        torch.set_num_threads(prior_threads)
