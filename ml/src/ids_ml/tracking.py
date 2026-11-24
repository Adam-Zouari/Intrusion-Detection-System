"""Local MLflow configuration and portable training-run provenance."""

from __future__ import annotations

import csv
import logging
import platform
import subprocess
import sys
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import mlflow
import psutil
from threadpoolctl import threadpool_info

from .data import find_project_root

_LOGGER = logging.getLogger(__name__)
_MIB = 1024**2


def tracking_uri(project_root: Path | None = None) -> str:
    root = project_root or find_project_root()
    return f"sqlite:///{(root / 'ml' / 'mlflow.db').resolve().as_posix()}"


def configure_tracking(project_root: Path | None = None) -> str:
    uri = tracking_uri(project_root)
    mlflow.set_tracking_uri(uri)
    return uri


def setup_mlflow_experiment(
    experiment_name: str, project_root: Path | None = None
) -> None:
    root = project_root or find_project_root()
    configure_tracking(root)
    artifact_root = root / "ml" / "mlruns"
    artifact_root.mkdir(parents=True, exist_ok=True)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(
            experiment_name, artifact_location=artifact_root.resolve().as_uri()
        )
    mlflow.set_experiment(experiment_name)


def _probe(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


@lru_cache(maxsize=8)
def _git_details(root: Path) -> tuple[str | None, str | None, bool | None]:
    git = ["git", "--no-optional-locks", "-C", str(root)]
    name = _probe([*git, "config", "--get", "user.name"]) or None
    commit = _probe([*git, "rev-parse", "--verify", "HEAD"]) or None
    status = _probe([*git, "status", "--porcelain"]) if commit else None
    return name, commit, bool(status) if status is not None else None


def _cpu_model() -> str | None:
    if sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    if sys.platform.startswith("linux"):
        try:
            with Path("/proc/cpuinfo").open(encoding="utf-8") as handle:
                for line in handle:
                    name, _, value = line.partition(":")
                    if name.strip() in {"model name", "Hardware"} and value.strip():
                        return value.strip()
        except OSError:
            pass
    return platform.processor() or None


def _available_gpus() -> list[dict[str, object]] | None:
    output = _probe([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if not output:
        return None
    devices: list[dict[str, object]] = []
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        if len(row) != 4:
            continue
        index, name, memory, driver = row
        devices.append({
            "index": int(index) if index.isdecimal() else None,
            "name": name,
            "total_vram_mib": int(memory) if memory.isdecimal() else None,
            "nvidia_driver_version": driver,
        })
    return devices or None


@lru_cache(maxsize=1)
def _hardware_environment() -> dict[str, object]:
    return {
        "operating_system": platform.platform(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_model": _cpu_model(),
        "physical_cpu_cores": psutil.cpu_count(logical=False),
        "logical_cpu_cores": psutil.cpu_count(),
        "total_ram_mib": psutil.virtual_memory().total / _MIB,
        "available_nvidia_gpus": _available_gpus(),
    }


def refresh_training_provenance() -> None:
    """Refresh metadata for a new CLI invocation in this process."""
    _git_details.cache_clear()
    _hardware_environment.cache_clear()


def start_training_run(
    *,
    run_name: str,
    source: str,
    tags: Mapping[str, str],
    project_root: Path | None = None,
) -> mlflow.ActiveRun:
    try:
        root = project_root or find_project_root()
    except FileNotFoundError:
        root = Path.cwd()
    user, commit, dirty = _git_details(root.resolve())
    provenance = {
        "mlflow.source.name": source,
        # MLflow's automatic Git context can include a credentialed remote URL.
        "mlflow.source.git.repoURL": "",
    }
    if user:
        provenance["mlflow.user"] = user
    if commit:
        provenance["mlflow.source.git.commit"] = commit
    if dirty is not None:
        provenance["git_worktree_dirty"] = str(dirty).lower()
    return mlflow.start_run(run_name=run_name, tags={**tags, **provenance})


def log_training_environment(
    *,
    training_device: str,
    inference_device: str,
    model_library: str | None = None,
    model_library_version: str | None = None,
    configured_threads: Mapping[str, int | None] | None = None,
) -> None:
    """Record actual devices separately from the hardware present on the host."""
    try:
        environment = {
            **_hardware_environment(),
            "training_device": training_device,
            "inference_device": inference_device,
            "model_library": model_library,
            "model_library_version": model_library_version,
            "configured_threads": dict(configured_threads or {}),
            "threadpools": [
                {
                    key: pool[key]
                    for key in ("user_api", "internal_api", "num_threads")
                    if key in pool
                }
                for pool in threadpool_info()
            ],
        }
        if model_library in {"pytorch-tabnet", "rtdl-revisiting-models"}:
            torch = sys.modules.get("torch")
            if torch is not None:
                environment["pytorch_version"] = torch.__version__
                environment["pytorch_cuda_build_version"] = torch.version.cuda
                environment["pytorch_threads"] = torch.get_num_threads()
        if model_library and model_library.lower() == "xgboost":
            xgb = sys.modules.get("xgboost")
            if xgb is not None:
                version = xgb.build_info().get("CUDA_VERSION")
                environment["xgboost_cuda_build_version"] = (
                    ".".join(map(str, version)) if isinstance(version, (list, tuple)) else version
                )
        gpu_devices = environment["available_nvidia_gpus"] or []
        tags = {
            "cpu_model": environment["cpu_model"],
            "gpu_model": ", ".join(gpu["name"] for gpu in gpu_devices),
            "machine": environment["architecture"],
            **{
                key: environment[key]
                for key in ("operating_system", "python_version", "model_library", "model_library_version")
            },
        }
        mlflow.set_tags({key: value for key, value in tags.items() if value})
        mlflow.log_dict(environment, "environment.json")
    except Exception as error:
        _LOGGER.warning("Training environment could not be recorded (%s).", type(error).__name__)
