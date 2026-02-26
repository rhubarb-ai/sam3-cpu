#!/usr/bin/env python3
"""
Example script demonstrating profiler control.

Usage:
    python profiler_example.py              # Profiling disabled (default)
    python profiler_example.py --profile    # Profiling enabled
"""

import sys
import time
import sam3.__globals
from sam3.utils.profiler import profile

# Check for --profile flag
if '--profile' in sys.argv:
    sam3.__globals.ENABLE_PROFILING = True
    print("🔍 Profiling ENABLED\n")
else:
    print("⚡ Profiling DISABLED (use --profile to enable)\n")


@profile()
def slow_computation():
    """Simulates a slow computation"""
    print("  Running slow computation...")
    time.sleep(0.5)
    result = sum(i**2 for i in range(1000000))
    return result


@profile()
def memory_intensive():
    """Simulates memory-intensive operation"""
    print("  Running memory-intensive operation...")
    data = [i for i in range(1000000)]
    time.sleep(0.2)
    return len(data)


@profile()
def quick_function():
    """A quick function"""
    print("  Running quick function...")
    time.sleep(0.1)
    return "quick result"


def main():
    print("=" * 60)
    print("Profiler Control Example")
    print("=" * 60)
    print(f"ENABLE_PROFILING = {sam3.__globals.ENABLE_PROFILING}\n")
    
    # Run functions
    print("1. Calling slow_computation()...")
    result1 = slow_computation()
    print(f"   ✓ Result: {result1}\n")
    
    print("2. Calling memory_intensive()...")
    result2 = memory_intensive()
    print(f"   ✓ Result: {result2}\n")
    
    print("3. Calling quick_function()...")
    result3 = quick_function()
    print(f"   ✓ Result: {result3}\n")
    
    print("=" * 60)
    if sam3.__globals.ENABLE_PROFILING:
        print("✓ Profiling complete!")
        print("  Check profile_results.json and profile_results.txt")
    else:
        print("✓ Done! (no profiling)")
        print("  Run with --profile flag to enable profiling")
    print("=" * 60)


if __name__ == "__main__":
    main()
