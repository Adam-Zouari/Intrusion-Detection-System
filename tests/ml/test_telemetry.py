from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import mlflow
import psutil
import pytest

from ids_ml.telemetry import monitor_training
from ids_ml.tracking import setup_mlflow_experiment

MIB = 1024**2


@pytest.fixture
def resource_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> mlflow.MlflowClient:
    original_uri = mlflow.get_tracking_uri()
    # Restore the caller's experiment choice without opening its store.
    monkeypatch.setattr(mlflow.tracking.fluent, "_active_experiment_id", None)
    setup_mlflow_experiment("resource-tests", tmp_path)
    try:
        yield mlflow.MlflowClient()
    finally:
        mlflow.set_tracking_uri(original_uri)


def test_resource_peaks_are_scoped_to_each_trial(
    resource_client: mlflow.MlflowClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient high-memory trial must not pollute the next trial in one process."""
    readings = SimpleNamespace(mib=100, target_mib=0, seen=threading.Event())

    def memory_info():
        mib = readings.mib
        if mib == readings.target_mib:
            readings.seen.set()
        return SimpleNamespace(rss=mib * MIB)

    for trial, peak in [(1, 250), (2, 110)]:
        readings.mib = 100
        readings.target_mib = peak
        readings.seen.clear()
        with mlflow.start_run(run_name=f"trial-{trial}") as run:
            with monkeypatch.context() as patch:
                patch.setattr(
                    psutil, "Process", lambda *_, **__: SimpleNamespace(memory_info=memory_info)
                )
                with monitor_training(
                    training_device="cpu", fit_scope="Shared matrices are resident."
                ):
                    readings.mib = peak
                    # Wait for a real sampler read, then return to baseline
                    # before exit. Boundary snapshots alone would miss this peak.
                    assert readings.seen.wait(timeout=5)
                    readings.mib = 100

        metrics = resource_client.get_run(run.info.run_id).data.metrics
        assert metrics["peak_training_ram_mib"] == peak
        assert "peak_training_vram_mib" not in metrics
        assert "peak_training_device_vram_mib" not in metrics
        artifact = resource_client.download_artifacts(
            run.info.run_id, "training_resources.json"
        )
        assert "Shared matrices are resident." in json.dumps(
            json.loads(Path(artifact).read_text(encoding="utf-8"))
        )


def test_fit_error_is_not_swallowed_and_sampler_stops(
    resource_client: mlflow.MlflowClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    readings = []

    def memory_info():
        readings.append(threading.current_thread())
        return SimpleNamespace(rss=37 * MIB)

    main_thread = threading.current_thread()
    sampler_seen = threading.Event()

    def notify_memory_info():
        info = memory_info()
        if threading.current_thread() is not main_thread:
            sampler_seen.set()
        return info

    with pytest.raises(ValueError, match="fit failed"):
        with mlflow.start_run(run_name="failed-fit") as run:
            with monkeypatch.context() as patch:
                patch.setattr(
                    psutil,
                    "Process",
                    lambda *_, **__: SimpleNamespace(memory_info=notify_memory_info),
                )
                with monitor_training(training_device="cpu"):
                    assert sampler_seen.wait(timeout=5)
                    raise ValueError("fit failed")

    result = resource_client.get_run(run.info.run_id)
    assert result.info.status == "FAILED"
    assert result.data.metrics["peak_training_ram_mib"] == 37
    samplers = {thread for thread in readings if thread is not main_thread}
    assert samplers and not any(thread.is_alive() for thread in samplers)


@pytest.mark.parametrize("process_usage", [None, 2**64 - 1, 384 * MIB])
def test_cuda_native_distinguishes_device_and_process_memory(
    resource_client: mlflow.MlflowClient,
    monkeypatch: pytest.MonkeyPatch,
    process_usage: int | None,
) -> None:
    """WDDM's unsupported value must not be reported as process memory or zero."""
    lifecycle = []
    fake_nvml = SimpleNamespace(
        nvmlInit=lambda: lifecycle.append("init"),
        nvmlShutdown=lambda: lifecycle.append("shutdown"),
        nvmlDeviceGetCount=lambda: 1,
        nvmlDeviceGetHandleByIndex=lambda _: "fake",
        nvmlDeviceGetMemoryInfo=lambda _: SimpleNamespace(used=640 * MIB, total=4096 * MIB),
        nvmlDeviceGetComputeRunningProcesses=lambda _: [
            SimpleNamespace(pid=-1, usedGpuMemory=99 * MIB),
            SimpleNamespace(pid=os.getpid(), usedGpuMemory=process_usage),
        ],
    )
    with mlflow.start_run(run_name="native-cuda") as run:
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "pynvml", fake_nvml)
            with monitor_training(training_device="cuda"):
                pass

    assert lifecycle == ["init", "shutdown"]
    saved = resource_client.get_run(run.info.run_id).data.metrics
    assert saved["peak_training_device_vram_mib"] == 640
    assert "peak_training_vram_reserved_mib" not in saved
    if process_usage is not None and process_usage <= 4096 * MIB:
        assert saved["peak_training_vram_mib"] == 384
    else:
        assert "peak_training_vram_mib" not in saved
        artifact = resource_client.download_artifacts(
            run.info.run_id, "training_resources.json"
        )
        resources = json.loads(Path(artifact).read_text(encoding="utf-8"))
        assert "per-process" in resources["notes"]["gpu_process"]


def test_torch_allocator_peaks_reset_without_nvml(
    resource_client: mlflow.MlflowClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    peaks = [900, 950]
    resets = []

    def reset(index):
        resets.append(index)
        peaks[:] = [0, 0]

    fake_cuda = SimpleNamespace(
        current_device=lambda: 0,
        reset_peak_memory_stats=reset,
        get_device_properties=lambda _: SimpleNamespace(uuid="GPU-mocked"),
        max_memory_allocated=lambda _: peaks[0] * MIB,
        max_memory_reserved=lambda _: peaks[1] * MIB,
    )
    for expected in [200, 75]:
        with mlflow.start_run(run_name="torch-cuda") as run:
            with monkeypatch.context() as patch:
                patch.setitem(sys.modules, "pynvml", None)
                patch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))
                with monitor_training(training_device="cuda", gpu_context="torch"):
                    # Allocator peak is maintained independently of sampling.
                    peaks[:] = [max(peaks[0], expected), max(peaks[1], expected + 50)]
        saved = resource_client.get_run(run.info.run_id).data.metrics
        assert saved["peak_training_vram_mib"] == expected
        assert saved["peak_training_vram_reserved_mib"] == expected + 50
        assert "peak_training_device_vram_mib" not in saved
    assert resets == [0, 0]
