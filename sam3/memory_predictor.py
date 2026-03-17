"""Adaptive hybrid memory predictor with OOM forecasting.

Models memory growth during frame-by-frame video inference as:

    M(f) = M₀ + k·f          (linear model)
    M(f) = M₀ · e^(k·f)      (exponential model)

Estimates **k** from a sliding window of near-zero cost measurements
(``torch.cuda.memory_allocated`` / ``psutil.Process.memory_info().rss``),
then predicts the OOM frame and triggers soft/hard stop signals before
the process crashes.

Architecture
------------
::

    FrameSample          – single measurement dataclass
    GrowthEstimator      – sliding-window O(1) growth-rate estimator
    HybridMemoryMonitor  – CPU (RSS) + per-GPU VRAM reader
    OOMPredictor         – prediction engine (linear + exponential)
    AdaptiveScheduler    – dynamic chunk-size optimiser
    MemoryPredictor      – async controller (daemon thread, callbacks)

Design Principles
-----------------
*  **Zero overhead** — uses only cached PyTorch counters and ``psutil``
   RSS reads; no ``pynvml``, no subprocess calls.
*  **Fully async** — background daemon thread polls at an exponentially
   decaying schedule.  Inference is never blocked.
*  **Hybrid** — tracks RAM (always) + VRAM (when CUDA is available).
   Works identically on CPU-only machines.
*  **Multi-GPU** — iterates ``torch.cuda.memory_allocated(dev)`` for
   each visible device.
*  **Growth detection** — classifies growth as linear / stepwise /
   exponential / fragmentation leak based on slope variance.
*  **Callbacks** — ``on_soft_stop``, ``on_hard_stop``, ``on_resize``
   hooks let the chunk system react before OOM.
"""

from __future__ import annotations

import math
import statistics
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any

import psutil

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_WINDOW = 30  # sliding window for k estimation
_DEFAULT_SAFETY = 0.85  # keep 15 % headroom before OOM
_MIN_SAMPLES_FOR_PRED = 1  # minimum slopes before we can predict (1 slope = 2 samples)
_SLOPE_VAR_LINEAR = 1e-6  # Var(k) below this → linear growth
_SLOPE_VAR_STEP = 1e-3  # Var(k) below this → stepwise
_CACHE_CLEAR_THRESHOLD = 0.90  # auto-clear CUDA cache at 90 % allocation


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class GrowthPattern(Enum):
    """Detected memory-growth classification."""

    UNKNOWN = auto()
    LINEAR = auto()
    STEPWISE = auto()
    EXPONENTIAL = auto()
    FRAGMENTATION = auto()


class StopLevel(Enum):
    """Urgency of an OOM warning."""

    NONE = auto()
    SOFT = auto()  # ~20 % headroom left
    HARD = auto()  # ~5 % headroom left
    CRITICAL = auto()  # OOM imminent (< 2 %)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FrameSample:
    """Snapshot of memory state after processing a frame."""

    frame: int
    timestamp: float  # time.monotonic()
    wall_time: str  # ISO timestamp
    rss_bytes: int  # process RSS
    gpu_allocated: dict[int, int] = field(default_factory=dict)
    gpu_reserved: dict[int, int] = field(default_factory=dict)


@dataclass
class Prediction:
    """OOM prediction result."""

    frames_remaining: int | None
    oom_frame: int | None
    confidence: float  # 0.0 – 1.0
    growth_rate_bytes: float  # k (bytes per frame)
    model_used: str  # "linear" | "exponential"
    memory_source: str  # "rss" | "gpu:0" etc.
    current_bytes: int
    limit_bytes: int
    headroom_pct: float
    stop_level: StopLevel


# ---------------------------------------------------------------------------
# HybridMemoryMonitor
# ---------------------------------------------------------------------------


class HybridMemoryMonitor:
    """Near-zero-cost memory reader for CPU + GPU.

    CPU:  ``psutil.Process().memory_info().rss``
    GPU:  ``torch.cuda.memory_allocated(dev)`` /
          ``torch.cuda.memory_reserved(dev)``

    Both are cached counters with negligible overhead.
    """

    def __init__(self):
        self._proc = psutil.Process()
        self._gpu_count = 0
        self._gpu_limits: dict[int, int] = {}
        self._torch_available = False

        try:
            import torch

            if torch.cuda.is_available():
                self._torch_available = True
                self._gpu_count = torch.cuda.device_count()
                for i in range(self._gpu_count):
                    # Prefer mem_get_info (reliable across PyTorch versions)
                    try:
                        _free, total = torch.cuda.mem_get_info(i)
                        self._gpu_limits[i] = total
                    except Exception:
                        props = torch.cuda.get_device_properties(i)
                        self._gpu_limits[i] = getattr(
                            props,
                            "total_memory",
                            getattr(props, "total_mem", 0),
                        )
        except Exception:
            pass

    # -- public API --------------------------------------------------------

    @property
    def gpu_count(self) -> int:
        return self._gpu_count

    @property
    def has_gpu(self) -> bool:
        return self._gpu_count > 0

    def rss_bytes(self) -> int:
        """Current process RSS in bytes."""
        return self._proc.memory_info().rss

    def ram_available(self) -> int:
        """System-wide available RAM in bytes."""
        return psutil.virtual_memory().available

    def ram_total(self) -> int:
        """System-wide total RAM in bytes."""
        return psutil.virtual_memory().total

    def gpu_allocated(self, device: int = 0) -> int:
        """VRAM currently allocated on *device* (bytes)."""
        if not self._torch_available or device >= self._gpu_count:
            return 0
        import torch

        return torch.cuda.memory_allocated(device)

    def gpu_reserved(self, device: int = 0) -> int:
        """VRAM reserved by the caching allocator on *device* (bytes)."""
        if not self._torch_available or device >= self._gpu_count:
            return 0
        import torch

        return torch.cuda.memory_reserved(device)

    def gpu_total(self, device: int = 0) -> int:
        """Total VRAM on *device* (bytes)."""
        return self._gpu_limits.get(device, 0)

    def gpu_free(self, device: int = 0) -> int:
        """Estimated free VRAM on *device* (bytes)."""
        total = self.gpu_total(device)
        if total == 0:
            return 0
        return total - self.gpu_allocated(device)

    def sample(self, frame: int) -> FrameSample:
        """Take a snapshot of all memory counters."""
        gpu_alloc = {}
        gpu_resv = {}
        for d in range(self._gpu_count):
            gpu_alloc[d] = self.gpu_allocated(d)
            gpu_resv[d] = self.gpu_reserved(d)

        return FrameSample(
            frame=frame,
            timestamp=time.monotonic(),
            wall_time=datetime.now().isoformat(),
            rss_bytes=self.rss_bytes(),
            gpu_allocated=gpu_alloc,
            gpu_reserved=gpu_resv,
        )

    def clear_gpu_cache(self, device: int | None = None):
        """Call ``torch.cuda.empty_cache()`` to release reserved memory."""
        if not self._torch_available:
            return
        import torch

        if device is not None:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        else:
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# GrowthEstimator  —  sliding-window O(1) growth-rate estimator
# ---------------------------------------------------------------------------


class GrowthEstimator:
    """Estimates memory growth rate *k* (bytes/frame) using a sliding
    window of recent samples.  Maintains running sums for O(1) update.

    Also classifies the growth pattern (linear / stepwise / exponential /
    fragmentation) by analysing the variance of per-step slopes.
    """

    def __init__(self, window: int = _DEFAULT_WINDOW):
        self._window = window
        self._frames: deque[int] = deque(maxlen=window)
        self._values: deque[int] = deque(maxlen=window)
        self._slopes: deque[float] = deque(maxlen=window)

    # ----- core -----------------------------------------------------------

    def add(self, frame: int, mem_bytes: int):
        """Record a new sample.  Automatically computes the incremental
        slope if a previous sample exists."""
        if self._frames and frame > self._frames[-1]:
            df = frame - self._frames[-1]
            dm = mem_bytes - self._values[-1]
            self._slopes.append(dm / df)
        self._frames.append(frame)
        self._values.append(mem_bytes)

    @property
    def k(self) -> float:
        """Current estimated growth rate (bytes per frame).

        Uses the **median** of the slope window to be robust against
        outliers (e.g. a single-frame GC spike).
        """
        if len(self._slopes) < 1:
            return 0.0
        return statistics.median(self._slopes)

    @property
    def k_mean(self) -> float:
        """Mean slope (less robust, but useful for comparisons)."""
        if len(self._slopes) < 1:
            return 0.0
        return statistics.mean(self._slopes)

    @property
    def slope_variance(self) -> float:
        """Variance of slopes in the current window."""
        if len(self._slopes) < 2:
            return 0.0
        return statistics.variance(self._slopes)

    @property
    def enough_data(self) -> bool:
        return len(self._slopes) >= _MIN_SAMPLES_FOR_PRED

    # ----- growth classification ------------------------------------------

    @property
    def pattern(self) -> GrowthPattern:
        """Classify the growth pattern from slope statistics."""
        if not self.enough_data:
            return GrowthPattern.UNKNOWN

        var = self.slope_variance
        k = self.k

        # Normalise variance relative to |k| to make thresholds scale-free
        if abs(k) > 1:
            norm_var = var / (k * k)
        else:
            norm_var = var

        if norm_var < _SLOPE_VAR_LINEAR:
            return GrowthPattern.LINEAR
        if norm_var < _SLOPE_VAR_STEP:
            return GrowthPattern.STEPWISE

        # Detect exponential: slopes should be *increasing*
        slopes = list(self._slopes)
        if len(slopes) >= 5:
            first_half = statistics.mean(slopes[: len(slopes) // 2])
            second_half = statistics.mean(slopes[len(slopes) // 2 :])
            if second_half > first_half * 1.3:
                return GrowthPattern.EXPONENTIAL

        return GrowthPattern.FRAGMENTATION

    # ----- exponential model fitting --------------------------------------

    def fit_exponential_k(self) -> float | None:
        """Estimate exponential growth rate using log-linear regression.

        If M(f) = M₀ · e^(k·f), then ln(M) = ln(M₀) + k·f, which is
        linear in f.  We compute k via simple linear regression on
        (frame, ln(mem)).
        """
        if len(self._frames) < _MIN_SAMPLES_FOR_PRED:
            return None
        frames = list(self._frames)
        vals = list(self._values)
        # Filter out zero/negative (can't take log)
        pairs = [(f, v) for f, v in zip(frames, vals) if v > 0]
        if len(pairs) < _MIN_SAMPLES_FOR_PRED:
            return None
        fs = [p[0] for p in pairs]
        lnm = [math.log(p[1]) for p in pairs]
        n = len(fs)
        mean_f = sum(fs) / n
        mean_ln = sum(lnm) / n
        num = sum((f - mean_f) * (l - mean_ln) for f, l in zip(fs, lnm))
        den = sum((f - mean_f) ** 2 for f in fs)
        if den == 0:
            return None
        return num / den

    # ----- state export ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "k_median": round(self.k, 2),
            "k_mean": round(self.k_mean, 2),
            "slope_variance": round(self.slope_variance, 4),
            "pattern": self.pattern.name,
            "window_size": self._window,
            "sample_count": len(self._frames),
        }


# ---------------------------------------------------------------------------
# OOMPredictor
# ---------------------------------------------------------------------------


class OOMPredictor:
    """Predicts OOM frame using linear (and optionally exponential) model.

    Linear:      f_oom = f_current + (M_limit − M_current) / k
    Exponential: f_oom = f_current + ln(M_limit / M_current) / k_exp
    """

    def __init__(self, safety_factor: float = _DEFAULT_SAFETY):
        self._safety = safety_factor

    def predict(
        self,
        current_frame: int,
        current_bytes: int,
        limit_bytes: int,
        estimator: GrowthEstimator,
        source_label: str = "rss",
    ) -> Prediction:
        """Generate a prediction for when OOM will occur."""
        effective_limit = int(limit_bytes * self._safety)
        headroom = (limit_bytes - current_bytes) / max(limit_bytes, 1)
        stop = self._stop_level(headroom)

        if not estimator.enough_data or estimator.k <= 0:
            return Prediction(
                frames_remaining=None,
                oom_frame=None,
                confidence=0.0,
                growth_rate_bytes=estimator.k,
                model_used="none",
                memory_source=source_label,
                current_bytes=current_bytes,
                limit_bytes=limit_bytes,
                headroom_pct=round(headroom * 100, 2),
                stop_level=stop,
            )

        # --- Linear prediction ---
        remaining_linear = (effective_limit - current_bytes) / estimator.k
        oom_linear = current_frame + max(int(remaining_linear), 0)

        # --- Exponential prediction (if pattern suggests it) ---
        model_used = "linear"
        frames_remaining = max(int(remaining_linear), 0)
        oom_frame = oom_linear

        if estimator.pattern == GrowthPattern.EXPONENTIAL:
            k_exp = estimator.fit_exponential_k()
            if k_exp and k_exp > 0 and current_bytes > 0:
                ratio = effective_limit / current_bytes
                if ratio > 0:
                    remaining_exp = math.log(ratio) / k_exp
                    if remaining_exp < remaining_linear:
                        frames_remaining = max(int(remaining_exp), 0)
                        oom_frame = current_frame + frames_remaining
                        model_used = "exponential"

        # Confidence based on data volume + slope stability
        n = len(estimator._slopes)
        confidence = min(1.0, n / (_DEFAULT_WINDOW * 2))
        if estimator.slope_variance > 0 and estimator.k != 0:
            cv = math.sqrt(estimator.slope_variance) / abs(estimator.k)
            confidence *= max(0.0, 1.0 - cv)

        return Prediction(
            frames_remaining=frames_remaining,
            oom_frame=oom_frame,
            confidence=round(confidence, 4),
            growth_rate_bytes=round(estimator.k, 2),
            model_used=model_used,
            memory_source=source_label,
            current_bytes=current_bytes,
            limit_bytes=limit_bytes,
            headroom_pct=round(headroom * 100, 2),
            stop_level=stop,
        )

    @staticmethod
    def _stop_level(headroom: float) -> StopLevel:
        if headroom > 0.20:
            return StopLevel.NONE
        if headroom > 0.05:
            return StopLevel.SOFT
        if headroom > 0.02:
            return StopLevel.HARD
        return StopLevel.CRITICAL


# ---------------------------------------------------------------------------
# AdaptiveScheduler  —  dynamic chunk-size optimiser
# ---------------------------------------------------------------------------


class AdaptiveScheduler:
    """Recommends an optimal chunk size based on observed memory growth.

    Strategy
    --------
    1. After processing ``warmup`` frames, measure M₀ and estimate k.
    2. Predict the maximum safe frame count:
       ``N = (M_limit − M₀) / k`` (with safety margin).
    3. At the midpoint (N/2), re-measure and recompute.
    4. Continue halving until the window is ≤ 2 frames or the prediction
       stabilises (change < 5 %).
    """

    def __init__(
        self,
        limit_bytes: int,
        safety: float = _DEFAULT_SAFETY,
        warmup: int = 3,
    ):
        self._limit = limit_bytes
        self._safety = safety
        self._warmup = warmup
        self._baseline: int | None = None
        self._recommended: int | None = None
        self._checkpoints: list[dict[str, Any]] = []

    @property
    def recommended_chunk(self) -> int | None:
        """Current best estimate of safe chunk size (frames)."""
        return self._recommended

    def update(self, frame: int, mem_bytes: int, estimator: GrowthEstimator):
        """Feed a new measurement; recalculates recommendation if needed."""
        if frame < self._warmup:
            return  # still warming up

        if self._baseline is None:
            self._baseline = mem_bytes

        k = estimator.k
        if k <= 0:
            return  # memory flat or shrinking — no OOM risk

        effective_limit = int(self._limit * self._safety)
        headroom = effective_limit - mem_bytes
        if headroom <= 0:
            self._recommended = frame  # already at limit
            return

        predicted_n = int(headroom / k)
        new_rec = frame + max(predicted_n, 1)

        self._checkpoints.append(
            {
                "frame": frame,
                "mem_bytes": mem_bytes,
                "k": round(k, 2),
                "predicted_safe_frames": predicted_n,
                "recommended_chunk": new_rec,
                "wall_time": datetime.now().isoformat(),
            }
        )

        # Accept new recommendation if it's more conservative or first
        if self._recommended is None or new_rec < self._recommended:
            self._recommended = new_rec

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_bytes": self._baseline,
            "recommended_chunk": self._recommended,
            "limit_bytes": self._limit,
            "safety_factor": self._safety,
            "checkpoints": self._checkpoints,
        }


# ---------------------------------------------------------------------------
# MemoryPredictor  —  main async controller
# ---------------------------------------------------------------------------


class MemoryPredictor:
    """Fully async, non-blocking memory predictor.

    Spawns a lightweight daemon thread that polls memory at an
    exponentially decaying schedule (frequent at start, tapering off).
    The inference loop simply calls :py:meth:`record_frame` after each
    frame — a near-zero-cost counter read.

    Parameters
    ----------
    device : str
        ``"cpu"`` or ``"cuda"`` or ``"cuda:N"``.
    safety_factor : float
        Fraction of the memory limit used as the effective ceiling
        (default: 0.85 → keeps 15 % headroom).
    window : int
        Sliding-window size for growth-rate estimation.
    poll_base : float
        Initial polling interval in seconds.
    poll_max : float
        Maximum polling interval (after exponential decay).
    on_soft_stop : callable
        Called when headroom drops below 20 %.
    on_hard_stop : callable
        Called when headroom drops below 5 %.
    on_resize : callable
        Called with ``(new_chunk_size: int)`` when the scheduler revises
        its recommendation downward.
    """

    def __init__(
        self,
        device: str = "cpu",
        safety_factor: float = _DEFAULT_SAFETY,
        window: int = _DEFAULT_WINDOW,
        poll_base: float = 0.5,
        poll_max: float = 5.0,
        on_soft_stop: Callable[[], None] | None = None,
        on_hard_stop: Callable[[], None] | None = None,
        on_resize: Callable[[int], None] | None = None,
    ):
        self._device_str = device
        self._safety = safety_factor
        self._poll_base = poll_base
        self._poll_max = poll_max

        # Callbacks
        self._on_soft = on_soft_stop
        self._on_hard = on_hard_stop
        self._on_resize = on_resize

        # Core components
        self._monitor = HybridMemoryMonitor()
        self._predictor = OOMPredictor(safety_factor)

        # Per-source estimators (one for RSS, one per GPU)
        self._estimators: dict[str, GrowthEstimator] = {
            "rss": GrowthEstimator(window),
        }
        for d in range(self._monitor.gpu_count):
            self._estimators[f"gpu:{d}"] = GrowthEstimator(window)

        # Determine limits
        self._limits: dict[str, int] = {
            "rss": self._monitor.ram_total(),
        }
        for d in range(self._monitor.gpu_count):
            self._limits[f"gpu:{d}"] = self._monitor.gpu_total(d)

        # Determine which source is the binding constraint
        if "cuda" in device and self._monitor.has_gpu:
            dev_idx = 0
            if ":" in device:
                try:
                    dev_idx = int(device.split(":")[1])
                except (ValueError, IndexError):
                    dev_idx = 0
            self._primary_source = f"gpu:{dev_idx}"
        else:
            self._primary_source = "rss"

        # Adaptive scheduler for the primary source
        primary_limit = self._limits[self._primary_source]
        self._scheduler = AdaptiveScheduler(primary_limit, safety_factor)

        # Sample history (full trace for metadata export)
        self._samples: list[FrameSample] = []  # from record_frame()
        self._bg_samples: list[FrameSample] = []  # from bg thread polls
        self._predictions: list[dict[str, Any]] = []
        self._last_prediction: Prediction | None = None

        # Background thread state
        self._running = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._poll_count = 0

        # Soft/hard stop already fired (avoid spamming callbacks)
        self._soft_fired = False
        self._hard_fired = False

    # =====================================================================
    # Public API — inference loop calls these
    # =====================================================================

    def start(self):
        """Start the background predictor thread."""
        self._running = True
        self._thread = threading.Thread(target=self._bg_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background thread and finalise metadata."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)

    def record_frame(self, frame: int):
        """Called by the inference loop after each frame.

        This is the *only* call in the hot path.  It reads cached memory
        counters (near-zero cost) and feeds them into the estimator.
        """
        sample = self._monitor.sample(frame)

        with self._lock:
            self._samples.append(sample)

            # Feed RSS estimator
            self._estimators["rss"].add(frame, sample.rss_bytes)

            # Feed GPU estimators
            for d, alloc in sample.gpu_allocated.items():
                key = f"gpu:{d}"
                if key in self._estimators:
                    self._estimators[key].add(frame, alloc)

            # Update scheduler
            primary_val = self._primary_value(sample)
            self._scheduler.update(frame, primary_val, self._primary_estimator)

    @property
    def prediction(self) -> Prediction | None:
        """Latest OOM prediction (thread-safe read)."""
        with self._lock:
            return self._last_prediction

    @property
    def should_stop(self) -> bool:
        """True if the predictor recommends halting inference."""
        pred = self.prediction
        if pred is None:
            return False
        return pred.stop_level in (StopLevel.HARD, StopLevel.CRITICAL)

    @property
    def recommended_chunk_size(self) -> int | None:
        """Current optimal chunk size estimate (frames)."""
        return self._scheduler.recommended_chunk

    # =====================================================================
    # Background thread
    # =====================================================================

    def _bg_loop(self):
        """Daemon loop: exponential backoff polling + headroom checks.

        The bg thread independently samples memory and checks headroom
        thresholds.  It does **not** feed the frame-based growth
        estimator (that only comes from ``record_frame()`` calls in the
        hot path) — it focuses on detecting danger levels from absolute
        memory values and triggering callbacks.
        """
        step = 0
        while self._running:
            # Exponential sampling schedule
            try:
                from sam3.__globals import POLL_BACKOFF_MAX_STEP
            except ImportError:
                POLL_BACKOFF_MAX_STEP = 50
            interval = min(
                self._poll_base * (1.2 ** min(step, POLL_BACKOFF_MAX_STEP)),
                self._poll_max,
            )
            time.sleep(interval)
            step += 1
            self._poll_count += 1

            with self._lock:
                # Take a bg sample for the growth curve (not for estimator)
                latest_frame = self._samples[-1].frame if self._samples else 0
                sample = self._monitor.sample(latest_frame)
                self._bg_samples.append(sample)

                # Run prediction using frame-based estimator data
                pred = self._make_prediction()
                if pred is not None:
                    self._last_prediction = pred
                    self._predictions.append(
                        {
                            "poll": self._poll_count,
                            "wall_time": datetime.now().isoformat(),
                            "source": pred.memory_source,
                            "current_bytes": pred.current_bytes,
                            "limit_bytes": pred.limit_bytes,
                            "headroom_pct": pred.headroom_pct,
                            "frames_remaining": pred.frames_remaining,
                            "oom_frame": pred.oom_frame,
                            "growth_rate": pred.growth_rate_bytes,
                            "model": pred.model_used,
                            "confidence": pred.confidence,
                            "stop_level": pred.stop_level.name,
                        }
                    )

                    # Fire callbacks
                    self._check_callbacks(pred)

                    # Auto CUDA cache clear if close to limit
                    if self._monitor.has_gpu and "gpu" in self._primary_source:
                        dev = int(self._primary_source.split(":")[1])
                        usage = self._monitor.gpu_allocated(dev) / max(self._monitor.gpu_total(dev), 1)
                        if usage > _CACHE_CLEAR_THRESHOLD:
                            self._monitor.clear_gpu_cache(dev)

    # =====================================================================
    # Internal helpers
    # =====================================================================

    @property
    def _primary_estimator(self) -> GrowthEstimator:
        return self._estimators[self._primary_source]

    def _primary_value(self, sample: FrameSample) -> int:
        if self._primary_source == "rss":
            return sample.rss_bytes
        dev = int(self._primary_source.split(":")[1])
        return sample.gpu_allocated.get(dev, 0)

    def _make_prediction(self) -> Prediction | None:
        """Run prediction on the primary memory source.

        Uses bg-polled sample for the most recent memory reading when
        available (finer temporal resolution), but frame count and
        growth rate *k* come from the frame-based estimator fed by
        ``record_frame()``.
        """
        if not self._samples:
            return None
        # Use whichever sample is most recent for current memory value
        latest = self._samples[-1]
        if self._bg_samples and self._bg_samples[-1].timestamp > latest.timestamp:
            latest = self._bg_samples[-1]
        current = self._primary_value(latest)
        limit = self._limits[self._primary_source]
        est = self._primary_estimator
        # Frame index always comes from frame-based samples
        frame = self._samples[-1].frame

        return self._predictor.predict(
            frame,
            current,
            limit,
            est,
            self._primary_source,
        )

    def _check_callbacks(self, pred: Prediction):
        if pred.stop_level == StopLevel.SOFT and not self._soft_fired:
            self._soft_fired = True
            if self._on_soft:
                try:
                    self._on_soft()
                except Exception:
                    pass
        if pred.stop_level in (StopLevel.HARD, StopLevel.CRITICAL) and not self._hard_fired:
            self._hard_fired = True
            if self._on_hard:
                try:
                    self._on_hard()
                except Exception:
                    pass
        # Resize callback when scheduler updates
        new_rec = self._scheduler.recommended_chunk
        if new_rec is not None and self._on_resize:
            try:
                self._on_resize(new_rec)
            except Exception:
                pass

    # =====================================================================
    # Metadata export
    # =====================================================================

    def summary(self) -> dict[str, Any]:
        """Full metadata export for JSON serialisation."""
        with self._lock:
            pred = self._last_prediction

            # Growth curves: downsample to max 200 points
            curve = self._build_growth_curve()

            return {
                "config": {
                    "device": self._device_str,
                    "safety_factor": self._safety,
                    "primary_source": self._primary_source,
                    "poll_base_s": self._poll_base,
                    "poll_max_s": self._poll_max,
                },
                "limits": {k: v for k, v in self._limits.items()},
                "total_samples": len(self._samples),
                "total_bg_samples": len(self._bg_samples),
                "total_polls": self._poll_count,
                "growth_estimators": {k: est.to_dict() for k, est in self._estimators.items()},
                "scheduler": self._scheduler.to_dict(),
                "last_prediction": {
                    "frames_remaining": pred.frames_remaining,
                    "oom_frame": pred.oom_frame,
                    "confidence": pred.confidence,
                    "growth_rate_bytes": pred.growth_rate_bytes,
                    "model_used": pred.model_used,
                    "headroom_pct": pred.headroom_pct,
                    "stop_level": pred.stop_level.name,
                }
                if pred
                else None,
                "prediction_history": self._predictions[-50:],  # last 50
                "growth_curve": curve,
            }

    def _build_growth_curve(self) -> dict[str, list]:
        """Build a downsampled growth curve for plotting / export.

        Merges frame-based samples from ``record_frame()`` with
        bg-polled samples for full temporal resolution, sorted by
        timestamp and deduped.
        """
        all_samples = list(self._samples) + list(self._bg_samples)
        if not all_samples:
            return {"frames": [], "rss_mb": [], "gpu_mb": {}, "timestamps": []}

        # Sort by timestamp, deduplicate by rounding ts to 0.01s
        all_samples.sort(key=lambda s: s.timestamp)
        seen_ts: set = set()
        unique: list[FrameSample] = []
        for s in all_samples:
            key = round(s.timestamp, 2)
            if key not in seen_ts:
                seen_ts.add(key)
                unique.append(s)

        # Downsample to ~200 points
        step = max(1, len(unique) // 200)
        frames = []
        rss_mb = []
        timestamps = []
        gpu_mb: dict[str, list[float]] = {}

        for i in range(0, len(unique), step):
            s = unique[i]
            frames.append(s.frame)
            timestamps.append(round(s.timestamp, 3))
            rss_mb.append(round(s.rss_bytes / (1024**2), 1))
            for d, alloc in s.gpu_allocated.items():
                key = f"gpu:{d}"
                if key not in gpu_mb:
                    gpu_mb[key] = []
                gpu_mb[key].append(round(alloc / (1024**2), 1))

        return {
            "frames": frames,
            "rss_mb": rss_mb,
            "gpu_mb": gpu_mb,
            "timestamps": timestamps,
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
