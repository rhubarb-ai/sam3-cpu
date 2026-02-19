import time
import tracemalloc
import functools
import json
import os
import psutil
from datetime import datetime
from sam3.__globals import PROFILE_OUTPUT_JSON, PROFILE_OUTPUT_TXT

PROFILE_RESULTS = []

def profile(output_json=PROFILE_OUTPUT_JSON, output_txt=PROFILE_OUTPUT_TXT):
    """Profile decorator with global enable/disable control via sam3.__globals.ENABLE_PROFILING"""

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check if profiling is enabled
            try:
                from sam3.__globals import ENABLE_PROFILING
            except ImportError:
                ENABLE_PROFILING = False
            
            if not ENABLE_PROFILING:
                # Profiling disabled - just call the function
                return func(*args, **kwargs)

            # Profiling enabled - measure performance
            process = psutil.Process(os.getpid())

            start_mem = process.memory_info().rss
            start_time = time.perf_counter()

            result = func(*args, **kwargs)

            end_time = time.perf_counter()
            end_mem = process.memory_info().rss

            execution_time = end_time - start_time
            memory_used = (end_mem - start_mem) / 10**6

            profile_data = {
                "function_name": func.__name__,
                "timestamp": datetime.now().isoformat(),
                "execution_time_seconds": round(execution_time, 6),
                "memory_used_MB": round(memory_used, 6),
                "total_process_memory_MB": round(end_mem / 10**6, 6)
            }

            PROFILE_RESULTS.append(profile_data)

            # Save results to JSON and TXT
            if output_json is not None:
                with open(output_json, "w") as f:
                    json.dump(PROFILE_RESULTS, f, indent=4)

            if output_txt is not None:
                with open(output_txt, "w") as f:
                    for entry in PROFILE_RESULTS:
                        f.write(
                            f"{entry['function_name']} | "
                        f"Time: {entry['execution_time_seconds']} s | "
                        f"Memory Used: {entry['memory_used_MB']} MB | "
                        f"Total Memory: {entry['total_process_memory_MB']} MB\n"
                    )

            print(f"[PROFILED] {func.__name__} | "
                  f"Time: {execution_time:.6f}s | "
                  f"Memory Used: {memory_used:.6f} MB")

            return result

        return wrapper

    return decorator

def profile_v1(output_json=PROFILE_OUTPUT_JSON, output_txt=PROFILE_OUTPUT_TXT):
    """
    Decorator to measure execution time and memory usage.
    Saves results to JSON and TXT.
    Uses tracemalloc for memory tracking.
    Controlled by sam3.__globals.ENABLE_PROFILING.
    """

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check if profiling is enabled
            try:
                from sam3.__globals import ENABLE_PROFILING
            except ImportError:
                ENABLE_PROFILING = False
            
            if not ENABLE_PROFILING:
                # Profiling disabled - just call the function
                return func(*args, **kwargs)

            # Profiling enabled - measure performance
            # Start tracking memory
            tracemalloc.start()

            start_time = time.perf_counter()

            result = func(*args, **kwargs)

            end_time = time.perf_counter()
            current, peak = tracemalloc.get_traced_memory()

            tracemalloc.stop()

            execution_time = end_time - start_time

            profile_data = {
                "function_name": func.__name__,
                "timestamp": datetime.now().isoformat(),
                "execution_time_seconds": round(execution_time, 6),
                "memory_current_MB": round(current / 10**6, 6),
                "memory_peak_MB": round(peak / 10**6, 6)
            }

            PROFILE_RESULTS.append(profile_data)

            # Save JSON
            if output_json is not None:
                with open(output_json, "w") as f:
                    json.dump(PROFILE_RESULTS, f, indent=4)

            # Save TXT
            if output_txt is not None:
                with open(output_txt, "w") as f:
                    for entry in PROFILE_RESULTS:
                        f.write(
                            f"Function: {entry['function_name']}\n"
                            f"Timestamp: {entry['timestamp']}\n"
                            f"Execution Time (s): {entry['execution_time_seconds']}\n"
                            f"Current Memory (MB): {entry['memory_current_MB']}\n"
                            f"Peak Memory (MB): {entry['memory_peak_MB']}\n"
                            f"{'-'*40}\n"
                        )

            print(f"[PROFILED] {func.__name__} | "
                  f"Time: {execution_time:.6f}s | "
                  f"Peak Memory: {peak / 10**6:.6f} MB")

            return result

        return wrapper

    return decorator