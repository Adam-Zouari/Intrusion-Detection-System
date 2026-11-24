"""Per-fit resource measurement. GPU libraries are never loaded on CPU fits."""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from threading import Event, Thread

import mlflow
import psutil

RESOURCE_METRIC_NAMES = (
    "peak_training_ram_mib", "peak_training_vram_mib",
    "peak_training_vram_reserved_mib", "peak_training_device_vram_mib",
)
_MIB = 1024**2
_INTERVAL = 0.1
_LOG = logging.getLogger(__name__)


class _FitSamples:
    def __init__(self, device: str, context: str):
        self.device = device
        self.context = context
        self.peaks: dict[str, float] = {}
        self.notes: dict[str, str] = {}
        self.process = None
        self.nvml = self.handle = self.cuda = None
        self.proc_supported = True
        self.cuda_index = 0

    def peak(self, key, size):
        if size is not None and size >= 0:
            self.peaks[key] = max(self.peaks.get(key, 0.0), size / _MIB)

    def start(self):
        try:
            self.process = psutil.Process(os.getpid())
        except (psutil.Error, OSError) as exc:
            self.notes["rss"] = type(exc).__name__
        if not self.device.startswith("cuda"):
            return
        uuid = None
        if self.context == "torch":
            try:
                cuda = sys.modules["torch"].cuda
                self.cuda_index = (int(self.device.split(":")[1]) if ":" in self.device
                                   else cuda.current_device())
                cuda.reset_peak_memory_stats(self.cuda_index)
                self.cuda = cuda
                uuid = getattr(cuda.get_device_properties(self.cuda_index), "uuid", None)
            except Exception as exc:
                self.notes["torch_allocator"] = type(exc).__name__
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            # Do not mistake a device-wide counter from another adapter for the
            # one used to train. A multi-GPU native run without a verifiable
            # UUID can still train; its GPU counters are simply unavailable.
            if pynvml.nvmlDeviceGetCount() == 1:
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            else:
                visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
                if not uuid and len(visible) > self.cuda_index:
                    uuid = visible[self.cuda_index].strip()
                if uuid and str(uuid).startswith(("GPU-", "MIG-")):
                    self.handle = pynvml.nvmlDeviceGetHandleByUUID(str(uuid))
                else:
                    self.notes["nvml"] = "Training device not unambiguously identified"
        except Exception as exc:
            self.notes["nvml"] = type(exc).__name__

    def sample(self):
        if self.process is not None:
            try:
                self.peak("peak_training_ram_mib", self.process.memory_info().rss)
            except (psutil.Error, OSError) as exc:
                self.notes["rss"] = type(exc).__name__
                self.process = None
        if self.handle is None:
            return
        try:
            memory = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
            self.peak("peak_training_device_vram_mib", memory.used)
            if self.context != "torch" and self.proc_supported:
                for p in self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle):
                    if p.pid != os.getpid():
                        continue
                    size = p.usedGpuMemory
                    if size is None or not 0 <= size <= memory.total:
                        self.notes["gpu_process"] = "Driver does not expose per-process usage"
                        self.proc_supported = False
                    else:
                        self.peak("peak_training_vram_mib", size)
                    break
        except Exception as exc:
            # Failure of process attribution shouldn't invalidate an otherwise
            # working device-wide counter (e.g. Windows WDDM).
            self.notes["nvml_sample"] = type(exc).__name__
            self.proc_supported = False

    def finish(self):
        self.sample()
        if self.cuda is not None:
            try:
                self.peak("peak_training_vram_mib", self.cuda.max_memory_allocated(self.cuda_index))
                self.peak("peak_training_vram_reserved_mib", self.cuda.max_memory_reserved(self.cuda_index))
            except Exception as exc:
                self.notes["torch_allocator"] = type(exc).__name__
        if self.nvml is not None:
            try:
                self.nvml.nvmlShutdown()
            except Exception as exc:
                self.notes["nvml_shutdown"] = type(exc).__name__
        if self.device.startswith("cuda") and "peak_training_vram_mib" not in self.peaks:
            self.notes.setdefault("gpu_process", "No attributable GPU memory measurement")


@contextmanager
def monitor_training(*, training_device: str, gpu_context: str = "native",
                     fit_scope: str = "Preprocessing and model fitting"):
    """Sample RSS and applicable GPU counters in this window, including on failure."""
    samples = _FitSamples(training_device, gpu_context)
    stop = Event()

    def poll():
        while not stop.wait(_INTERVAL):
            samples.sample()

    samples.start()
    samples.sample()
    worker = Thread(target=poll, name="ids-training-resources", daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join()
        samples.finish()
        # Smoke tests may use the underlying estimators without an MLflow run.
        if mlflow.active_run() is not None:
            try:
                if samples.peaks:
                    mlflow.log_metrics(samples.peaks, synchronous=True)
                mlflow.log_dict({
                    "scope": fit_scope,
                    "sample_interval_seconds": _INTERVAL,
                    "ram_method": "Process RSS including resident inputs; excludes children",
                    "vram_method": (
                        "not_applicable" if not training_device.startswith("cuda") else
                        "torch_allocator" if gpu_context == "torch" else "nvml_sampled_process"
                    ),
                    "device_vram_method": "sampled NVML device-wide usage (all processes)"
                    if training_device.startswith("cuda") else "not_applicable",
                    "metrics_mib": samples.peaks,
                    "notes": samples.notes,
                }, "training_resources.json")
            except Exception:
                _LOG.warning("Could not log training resource measurements", exc_info=True)
