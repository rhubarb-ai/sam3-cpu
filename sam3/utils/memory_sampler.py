"""
Lightweight background memory sampler.

Polls process RSS (and optionally GPU memory) at regular intervals,
recording peak and minimum values with timestamps.  Used to track
memory usage during inference without significant overhead.
"""

import threading
import time
from datetime import datetime


class MemorySampler:
    """Background thread that polls process memory at regular intervals.

    Args:
        interval: Seconds between polls (default: 1.0).
        device: ``"cpu"`` or ``"cuda"`` — controls whether GPU stats
            are captured on :py:meth:`stop`.
    """

    def __init__(self, interval: float = 1.0, device: str = "cpu"):
        import psutil

        self._proc = psutil.Process()
        self._interval = interval
        self._device = device
        self._running = False
        self._thread: threading.Thread | None = None

        # Tracking state
        self._peak_rss = 0
        self._peak_ts: str | None = None
        self._min_rss = float("inf")
        self._min_ts: str | None = None

    # ------------------------------------------------------------------
    def start(self):
        """Begin polling in a daemon thread."""
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the polling thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)

    # ------------------------------------------------------------------
    def _poll(self):
        while self._running:
            try:
                rss = self._proc.memory_info().rss
                ts = datetime.now().isoformat()
                if rss > self._peak_rss:
                    self._peak_rss = rss
                    self._peak_ts = ts
                if rss < self._min_rss:
                    self._min_rss = rss
                    self._min_ts = ts
            except Exception:
                pass  # process might be exiting
            time.sleep(self._interval)

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Return a dict summarising peak / min memory observations."""
        result = {
            "peak_rss_bytes": self._peak_rss,
            "peak_rss_timestamp": self._peak_ts,
            "min_rss_bytes": self._min_rss if self._min_rss != float("inf") else None,
            "min_rss_timestamp": self._min_ts,
            "gpu_peak_allocated_bytes": None,
        }
        if self._device == "cuda":
            try:
                import torch

                if torch.cuda.is_available():
                    result["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            except Exception:
                pass
        return result
