import torch
import os
import sys
import sam3
from sam3.logger import get_logger

logger = get_logger(__name__)

logger.info(f"Python executable: {sys.executable}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check for --profile flag
if '--profile' in sys.argv:
    ENABLE_PROFILING = True
    logger.info("🔍 Profiling ENABLED\n")
else:
    ENABLE_PROFILING = False
    logger.info("⚡ Profiling DISABLED (use --profile to enable)\n")

SAM3_ROOT = os.path.join(os.path.dirname(sam3.__file__))
BPE_PATH = os.path.join(SAM3_ROOT, "assets/bpe_simple_vocab_16e6.txt.gz")

# Video processing defaults
DEFAULT_MIN_VIDEO_FRAMES = 15
DEFAULT_MIN_CHUNK_OVERLAP = 1

# Memory management
IMAGE_INFERENCE_MB = 6760
VIDEO_INFERENCE_MB = 6900
RAM_USAGE_PERCENT = 0.25  # Use 25% of available RAM for video chunking
MEMORY_SAFETY_MULTIPLIER = 3  # Require 3x estimated memory for safety
CPU_MEMORY_RESERVE_PERCENT = 0.3  # Reserve 30% for OS
GPU_MEMORY_RESERVE_PERCENT = 0.05  # Reserve 5% for display

# Parallel processing
PARALLEL_CHUNK_THRESHOLD = 0.90  # Start loading next chunk at 90% completion

# Output settings
DEFAULT_PROPAGATION_DIRECTION = "both"
DEFAULT_NUM_WORKERS = 1
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# Directory settings
TEMP_DIR = "/tmp/sam3-cpu" if DEVICE.type == "cpu" else "/tmp/sam3-gpu"
os.makedirs(TEMP_DIR, exist_ok=True)

DEFAULT_OUTPUT_DIR = os.path.join("./results")
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)