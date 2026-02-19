import argparse
import subprocess
import json
import os
import math
import re
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

def sanitize_filename(filename: str, replacement: str = "-") -> str:
    """
    Sanitize a string to make it safe for use as a filename or directory name.
    
    This function:
    - Replaces spaces with the specified replacement character (default: "-")
    - Removes or replaces invalid characters for filesystems
    - Ensures the result is a valid filename
    
    Args:
        filename: The string to sanitize (e.g., a prompt with spaces).
        replacement: Character to replace spaces and invalid chars with. Defaults to "-".
    
    Returns:
        Sanitized filename safe for use in file/directory names.
    
    Example:
        >>> sanitize_filename("person walking on street")
        'person-walking-on-street'
        >>> sanitize_filename("object: car/truck")
        'object-car-truck'
    """
    # Replace spaces with replacement character
    filename = filename.replace(" ", replacement)
    
    # Remove or replace invalid characters for filesystems
    # Invalid chars: < > : " / \ | ? * and control characters
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', replacement, filename)
    
    # Remove leading/trailing dots and spaces (Windows compatibility)
    filename = filename.strip('. ')
    
    # Replace multiple consecutive replacement characters with single one
    filename = re.sub(f'{re.escape(replacement)}+', replacement, filename)
    
    # Remove leading/trailing replacement characters
    filename = filename.strip(replacement)
    
    # Ensure filename is not empty
    if not filename:
        filename = "unnamed"
    
    # Limit length to 255 characters (common filesystem limit)
    if len(filename) > 255:
        filename = filename[:255].rstrip(replacement)
    
    return filename

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
