import argparse
import subprocess
import json
import os
import math
import psutil
from pathlib import Path

# ============================================================
# Helpers
# ============================================================

def run_cmd(cmd):
    """Run command and return stdout."""
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()

def vram_stat():
    """Get VRAM usage statistics using nvidia-smi."""
    try:
        output = run_cmd([
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits"
        ])
        total, used, free = map(int, output.split(","))
        return {
            'total': total * 1024 * 1024,
            'used': used * 1024 * 1024,
            'free': free * 1024 * 1024,
            'percent': (used / total) * 100 if total > 0 else 0
        }
    except Exception as e:
        raise RuntimeError(f"Failed to get VRAM stats: {str(e)}")

def ram_stat():
    """Get RAM usage statistics."""
    mem = psutil.virtual_memory()
    return {
        'total': mem.total,
        'available': mem.available,
        'used': mem.used,
        'percent': mem.percent
    }
