"""Isolated SQLite checks for portable MLflow provenance and environment records."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pytest

from ids_ml import tracking


@pytest.mark.parametrize("git_name", ["Adam Zouari", None])
def test_creator_source_and_environment(
    git_name: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def git_probe(command: list[str]):
        if command[-3:] == ["config", "--get", "user.name"]:
            return git_name
        if command[-3:] == ["rev-parse", "--verify", "HEAD"]:
            return "a" * 40
        if command[-2:] == ["status", "--porcelain"]:
            return " M pyproject.toml"
        return None  # Simulate a machine with no NVIDIA GPU or driver probe.

    monkeypatch.setattr(tracking, "_probe", git_probe)
    tracking.refresh_training_provenance()
    previous_uri = mlflow.get_tracking_uri()
    tracking.setup_mlflow_experiment("isolated-provenance-test", tmp_path)
    try:
        with mlflow.start_run() as default_run:
            default_user = default_run.info.user_id
        with tracking.start_training_run(
            run_name="portable-test",
            source="ids-tune-trees compare-original-xgboost",
            tags={"purpose": "synthetic"},
            project_root=tmp_path,
        ) as active:
            tracking.log_training_environment(
                training_device="cpu",
                inference_device="cpu",
                model_library="scikit-learn",
                model_library_version="test-version",
                configured_threads={"n_jobs": -1},
            )
            run_id = active.info.run_id
        client = mlflow.MlflowClient()
        run = client.get_run(run_id)
        assert run.info.user_id == (git_name or default_user)
        assert run.data.tags["mlflow.user"] == (git_name or default_user)
        assert run.data.tags["mlflow.source.name"] == "ids-tune-trees compare-original-xgboost"
        assert run.data.tags["mlflow.source.git.commit"] == "a" * 40
        assert run.data.tags["git_worktree_dirty"] == "true"
        artifact_path = client.download_artifacts(run_id, "environment.json")
        artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        assert artifact["training_device"] == artifact["inference_device"] == "cpu"
        assert artifact["available_nvidia_gpus"] is None
        assert artifact["total_ram_mib"] > 0
        assert artifact["configured_threads"]["n_jobs"] == -1
        assert not any("filepath" in pool for pool in artifact["threadpools"])
    finally:
        mlflow.end_run()
        mlflow.set_tracking_uri(previous_uri)
        tracking.refresh_training_provenance()
