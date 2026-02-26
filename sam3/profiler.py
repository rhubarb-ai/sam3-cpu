"""Backward-compatible re-export — moved to sam3.utils.profiler."""
from sam3.utils.profiler import profile, get_profiling_results, clear_profiling_results

__all__ = ["profile", "get_profiling_results", "clear_profiling_results"]
